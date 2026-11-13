// SPDX-License-Identifier: GPL-2.0-or-later
// The render adapter's Metal presents (mtl_present.h). ARC.

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "video_core/renderer_metal/mtl_frame.h"
#include "video_core/renderer_metal/mtl_present.h"
#include "video_core/renderer_metal/mtl_vk_layer.h"

namespace MetalPresent {

void Configure(void* layer_ptr, const LayerConfig& config) {
    CAMetalLayer* layer = (__bridge CAMetalLayer*)layer_ptr;
    layer.device = (__bridge id<MTLDevice>)MetalLayer::MtlDevice();
    // Presents copy into the drawable (a blit), so its texture is not framebuffer-only.
    layer.framebufferOnly = NO;
    layer.maximumDrawableCount = std::clamp<NSUInteger>(config.max_drawables, 2, 3);
#if !TARGET_OS_IPHONE // iOS always syncs to the display
    layer.displaySyncEnabled = config.display_sync;
#endif
    layer.opaque = YES;
    CGColorSpaceRef space = CGColorSpaceCreateWithName(
        config.hdr ? kCGColorSpaceITUR_2100_PQ : kCGColorSpaceSRGB);
    layer.colorspace = space;
    CGColorSpaceRelease(space);
    layer.pixelFormat = config.hdr ? MTLPixelFormatRGB10A2Unorm : MTLPixelFormatBGRA8Unorm;
    layer.wantsExtendedDynamicRangeContent = config.hdr;
    // Apple's Metal performance HUD on the game's layer (iOS 16 / macOS 13). MTL_HUD_ENABLED=1 in
    // the environment alone is read when Metal starts, which in an app that drew its own screen
    // before the game (the iOS launcher) is too early: the HUD then never showed.
    if (const char* hud = std::getenv("MTL_HUD_ENABLED"); hud && hud[0] == '1') {
        if (@available(iOS 16.0, macOS 13.0, *)) {
            if ([layer respondsToSelector:@selector(setDeveloperHUDProperties:)]) {
                layer.developerHUDProperties = @{@"mode" : @"default"};
                std::printf("Metal HUD: on the game's layer\n");
            }
        }
    }
}

std::pair<u32, u32> FitDrawableSize(void* layer_ptr) {
    CAMetalLayer* layer = (__bridge CAMetalLayer*)layer_ptr;
    const CGSize bounds = layer.bounds.size;
    const CGFloat scale = layer.contentsScale;
    const u32 width = u32(std::max<CGFloat>(0, bounds.width * scale));
    const u32 height = u32(std::max<CGFloat>(0, bounds.height * scale));
    if (width && height) {
        layer.drawableSize = CGSizeMake(width, height);
    }
    return {width, height};
}

bool DisplaySupportsHdr(void* layer_ptr) {
    (void)layer_ptr;
    // Every display macOS drives can take extended-range content; whether it shows more than SDR
    // is the display's (as MoltenVK reports the HDR10 surface format on any Mac).
    return true;
}

namespace {

/// The present pass: a full-screen triangle sampling the frame (linear, as the Vulkan
/// presenter's blit filtered), drawn into the letterbox rectangle (the viewport).
constexpr const char* PresentSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Out {
    float4 position [[position]];
    float2 uv;
};
vertex Out present_vertex(uint id [[vertex_id]]) {
    const float2 uv = float2((id << 1) & 2, id & 2);
    Out out;
    out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}
fragment float4 present_fragment(Out in [[stage_in]], texture2d<float> frame [[texture(0)]],
                                 sampler linear [[sampler(0)]]) {
    return frame.sample(linear, in.uv);
}
)";

struct Passes {
    id<MTLLibrary> library;
    id<MTLSamplerState> sampler;
    std::mutex mutex;
    std::unordered_map<NSUInteger, id<MTLRenderPipelineState>> pipelines; ///< by pixel format
};

Passes& GetPasses(id<MTLDevice> device) {
    static Passes* passes = [&] {
        auto* p = new Passes; // lives with the process, as the device
        NSError* error = nil;
        p->library = [device newLibraryWithSource:@(PresentSource) options:nil error:&error];
        if (!p->library) {
            std::printf("Metal present: the present shader did not compile: %s\n",
                        error.localizedDescription.UTF8String);
        }
        MTLSamplerDescriptor* sampler = [MTLSamplerDescriptor new];
        sampler.minFilter = MTLSamplerMinMagFilterLinear;
        sampler.magFilter = MTLSamplerMinMagFilterLinear;
        sampler.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sampler.tAddressMode = MTLSamplerAddressModeClampToEdge;
        p->sampler = [device newSamplerStateWithDescriptor:sampler];
        return p;
    }();
    return *passes;
}

id<MTLRenderPipelineState> Pipeline(Passes& passes, id<MTLDevice> device, MTLPixelFormat format) {
    std::scoped_lock lock{passes.mutex};
    if (auto it = passes.pipelines.find(format); it != passes.pipelines.end()) {
        return it->second;
    }
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [passes.library newFunctionWithName:@"present_vertex"];
    desc.fragmentFunction = [passes.library newFunctionWithName:@"present_fragment"];
    desc.colorAttachments[0].pixelFormat = format;
    NSError* error = nil;
    id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:desc
                                                                                 error:&error];
    if (!pipeline) {
        std::printf("Metal present: no present pipeline: %s\n",
                    error.localizedDescription.UTF8String);
    }
    passes.pipelines.emplace(format, pipeline);
    return pipeline;
}


/// Frame generation (BB_FRAMEGEN=1): MetalFX's frame interpolator makes the frame halfway between
/// each two presents, shown one 60 Hz period before the game's own (the game at 30 FPS, the screen
/// at 60). Adapted from cheemu21/kid-with-sword-remake (an nxnative fork, MIT; a758ea2 and
/// c73064e, dm/gfx/nvn/nvn_mtl.m fg_for/fg_encode/fg_motion/fg_keep and dm/gfx/msl/fgmv.metal):
/// - the motion between the two pictures is searched here at 1/8 size (64 px tiles within
///   +-192 px), so fast camera turns are placed exactly (BB_FRAMEGEN_MV=0: MetalFX's own flow);
/// - still detailed pixels (the HUD) are taken from the game's frame after the interpolation;
/// - a late frame shows no generated frame, and while frames keep dropping the interpolation
///   stops for 30 presents, doubling up to 300, so it does not flap.
/// The depth is constant (far, reversed).
constexpr const char* FgmvSource = R"FGMV(
// Frame generation's motion (nvn_mtl.m fg_motion): the motion between two presented pictures, estimated here because the game gives none, as the MetalFX frame interpolator's
// motion texture. MetalFX's own flow finds motion up to about 64 px a frame at 1176x540 (and tears or gives up past 96-192 px: the picture stays on one of the two frames, a
// judder on fast camera turns); with motion vectors it places any pan exactly. Three passes at 1/8 of the picture's size:
//   fgmv_luma    each 8x8 block's mean luma (one per present, kept with its ring slot for the next present's search)
//   fgmv_search  each tile of 8x8 of those (64x64 px of the picture) searched against the previous present's luma within +-FGMV_R (+-192 px), with a sub-pixel fit
//   fgmv_pick    each 1/8 pixel takes, of its tile's vector, its 8 neighbours' and no motion, the one that matches best around it, so a still HUD or a character the camera
//                follows keeps no motion beside the panning background
// The motion texture is at 1/8 size, its vectors in its own pixels pointing to where a pixel was in the previous picture (MetalFX's convention, scale 1)
#include <metal_stdlib>
using namespace metal;

#define FGMV_T 8                       // tile size in 1/8 pixels
#define FGMV_R 24                      // search radius in 1/8 pixels
#define FGMV_P (FGMV_T + 2 * FGMV_R)   // the previous picture's window a tile searches
#define FGMV_N ((2 * FGMV_R + 1) * (2 * FGMV_R + 1))
#define FGMV_TG 256

kernel void fgmv_luma(texture2d<float, access::read> src [[texture(0)]], texture2d<half, access::write> dst [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= dst.get_width() || g.y >= dst.get_height()) return;
    uint2 mx = uint2(src.get_width() - 1, src.get_height() - 1), o = g * 8;
    float s = 0;
    for (uint y = 0; y < 8; y += 2) for (uint x = 0; x < 8; x += 2) {   // 16 of the 64: the mean of a block whose detail the search does not need
        float3 c = src.read(min(o + uint2(x, y), mx)).rgb;
        s += dot(c, float3(0.299, 0.587, 0.114));
    }
    dst.write(half4(half(s / 16)), g);
}

// tiles: xy the vector (1/8 pixels, pointing to the previous picture), z how far its match beats the tile's mean (0 = a flat tile, nothing to go by)
kernel void fgmv_search(texture2d<half, access::read> cur [[texture(0)]], texture2d<half, access::read> prev [[texture(1)]], texture2d<half, access::write> tiles [[texture(2)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint li [[thread_index_in_threadgroup]]) {
    threadgroup half c[FGMV_T * FGMV_T];
    threadgroup half p[FGMV_P * FGMV_P];
    threadgroup float bc[FGMV_TG], bs[FGMV_TG], sm[FGMV_TG];
    threadgroup int bk[FGMV_TG];
    int2 mx = int2(cur.get_width() - 1, cur.get_height() - 1), t0 = int2(tg) * FGMV_T;
    for (uint i = li; i < FGMV_T * FGMV_T; i += FGMV_TG) c[i] = cur.read(uint2(clamp(t0 + int2(i % FGMV_T, i / FGMV_T), int2(0), mx))).r;
    for (uint i = li; i < FGMV_P * FGMV_P; i += FGMV_TG) p[i] = prev.read(uint2(clamp(t0 - FGMV_R + int2(i % FGMV_P, i / FGMV_P), int2(0), mx))).r;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float best = INFINITY, bsad = 0, sum = 0; int bkk = FGMV_N / 2;
    for (int k = (int)li; k < FGMV_N; k += FGMV_TG) {
        int ox = k % (2 * FGMV_R + 1), oy = k / (2 * FGMV_R + 1);   // window offsets (FGMV_R, FGMV_R = no motion)
        float sad = 0;
        for (int y = 0; y < FGMV_T; y++) for (int x = 0; x < FGMV_T; x++) sad += fabs(float(c[y * FGMV_T + x]) - float(p[(y + oy) * FGMV_P + x + ox]));
        sad *= 1.0f / (FGMV_T * FGMV_T);
        float cost = sad + 0.0004f * float(abs(ox - FGMV_R) + abs(oy - FGMV_R));   // ties (repeated patterns, flat areas) go to the smaller motion
        sum += sad;
        if (cost < best) { best = cost; bsad = sad; bkk = k; }
    }
    bc[li] = best; bs[li] = bsad; bk[li] = bkk; sm[li] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = FGMV_TG / 2; s > 0; s >>= 1) {
        if (li < s) {
            if (bc[li + s] < bc[li] || (bc[li + s] == bc[li] && bk[li + s] < bk[li])) { bc[li] = bc[li + s]; bs[li] = bs[li + s]; bk[li] = bk[li + s]; }
            sm[li] += sm[li + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (li) return;
    int ox = bk[0] % (2 * FGMV_R + 1), oy = bk[0] / (2 * FGMV_R + 1);
    float2 v = float2(ox - FGMV_R, oy - FGMV_R);
    // sub-pixel: a parabola through the costs beside the best one
    if (ox > 0 && ox < 2 * FGMV_R && oy > 0 && oy < 2 * FGMV_R) {
        float s0 = bs[0], sx[2], sy[2];
        for (int d = 0; d < 2; d++) {
            int dd = d ? 1 : -1; float a = 0, b = 0;
            for (int y = 0; y < FGMV_T; y++) for (int x = 0; x < FGMV_T; x++) {
                float cv = float(c[y * FGMV_T + x]);
                a += fabs(cv - float(p[(y + oy) * FGMV_P + x + ox + dd])); b += fabs(cv - float(p[(y + oy + dd) * FGMV_P + x + ox]));
            }
            sx[d] = a * (1.0f / (FGMV_T * FGMV_T)); sy[d] = b * (1.0f / (FGMV_T * FGMV_T));
        }
        float qx = sx[0] - 2 * s0 + sx[1], qy = sy[0] - 2 * s0 + sy[1];
        if (qx > 1e-5f) v.x += clamp(0.5f * (sx[0] - sx[1]) / qx, -0.5f, 0.5f);
        if (qy > 1e-5f) v.y += clamp(0.5f * (sy[0] - sy[1]) / qy, -0.5f, 0.5f);
    }
    float mean = sm[0] / FGMV_N, conf = mean > 1e-3f ? clamp((mean - bs[0]) / mean, 0.0f, 1.0f) : 0.0f;
    tiles.write(half4(half(v.x), half(v.y), half(conf), 0), tg);
}

// The 3x3 window's mean absolute difference between cur at g and prev at g + v (bilinear, in 1/8 pixels)
static float fgmv_cost(texture2d<half, access::read> cur, texture2d<half, access::sample> prev, int2 g, float2 v, float2 inv, int2 mx) {
    constexpr sampler ls(coord::normalized, address::clamp_to_edge, filter::linear);
    float s = 0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        int2 q = clamp(g + int2(x, y), int2(0), mx);
        s += fabs(float(cur.read(uint2(q)).r) - float(prev.sample(ls, (float2(q) + 0.5f + v) * inv).r));
    }
    return s * (1.0f / 9);
}
kernel void fgmv_pick(texture2d<half, access::read> cur [[texture(0)]], texture2d<half, access::sample> prev [[texture(1)]], texture2d<half, access::read> tiles [[texture(2)]],
                      texture2d<half, access::write> mot [[texture(3)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= mot.get_width() || g.y >= mot.get_height()) return;
    int2 mx = int2(cur.get_width() - 1, cur.get_height() - 1), tm = int2(tiles.get_width() - 1, tiles.get_height() - 1), t = int2(g) / FGMV_T, gi = int2(g);
    float2 inv = 1.0f / float2(prev.get_width(), prev.get_height());
    float2 bv = 0; float bc = fgmv_cost(cur, prev, gi, 0, inv, mx) - 0.004f;   // no motion wins ties: a still HUD and flat areas stay put
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        half4 tv = tiles.read(uint2(clamp(t + int2(x, y), int2(0), tm)));
        if (tv.z < 0.05h) continue;   // a flat tile's vector says nothing
        float2 v = float2(tv.xy); float cst = fgmv_cost(cur, prev, gi, v, inv, mx) - (x == 0 && y == 0 ? 0.001f : 0);
        if (cst < bc) { bc = cst; bv = v; }
    }
    mot.write(half4(half(bv.x), half(bv.y), 0, 0), g);
}

// The HUD (nvn_mtl.m fg_keep): with motion anywhere in the picture MetalFX moves still pixels beside it too, so a still HUD over a panning background smears even when its
// own vectors say it stays (probe: 0.03-0.09 of mean error on the HUD, against 0.0015 with no motion at all). Pixels that did not change since the previous present and have
// detail around them (a HUD, the character the camera follows when it stands still), and still pixels near those (a HUD's flat fill), are taken from the present as it is.
// A flat still area alone is left to the interpolator: something thin that crossed it between the two presents is in the generated frame there.
//   fgmv_still  per 4x4 block of the picture: 1 still, 2 still with detail (a 6x6 window: no channel changed by more than 2/255; luma spread over 0.08)
//   fgmv_hud    per block: still, with a still detailed block within 3 blocks (12 px)
//   fgmv_keep   per pixel of the generated frame: the present's own where its block is kept
kernel void fgmv_still(texture2d<float, access::read> cur [[texture(0)]], texture2d<float, access::read> prev [[texture(1)]], texture2d<half, access::write> st [[texture(2)]],
                       uint2 g [[thread_position_in_grid]]) {
    if (g.x >= st.get_width() || g.y >= st.get_height()) return;
    int2 mx = int2(cur.get_width() - 1, cur.get_height() - 1), o = int2(g) * 4 - 1;
    float d = 0, lo = 1e9, hi = -1e9;
    for (int y = 0; y < 6; y++) for (int x = 0; x < 6; x++) {
        uint2 p = uint2(clamp(o + int2(x, y), int2(0), mx)); float3 a = cur.read(p).rgb, b = prev.read(p).rgb;
        float3 e = fabs(a - b); d = max(d, max(e.x, max(e.y, e.z)));
        float l = dot(a, float3(0.299, 0.587, 0.114)); lo = min(lo, l); hi = max(hi, l);
    }
    half v = d <= 2.5h / 255 ? (hi - lo > 0.08 ? 2.0h : 1.0h) : 0.0h;
    st.write(half4(v), g);
}
kernel void fgmv_hud(texture2d<half, access::read> st [[texture(0)]], texture2d<half, access::write> keep [[texture(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= keep.get_width() || g.y >= keep.get_height()) return;
    int2 mx = int2(st.get_width() - 1, st.get_height() - 1), gi = int2(g);
    half k = 0;
    if (st.read(g).r > 0.5h)
        for (int y = -3; y <= 3 && k == 0; y++) for (int x = -3; x <= 3; x++) if (st.read(uint2(clamp(gi + int2(x, y), int2(0), mx))).r > 1.5h) { k = 1; break; }
    keep.write(half4(k), g);
}
kernel void fgmv_keep(texture2d<float, access::read> cur [[texture(0)]], texture2d<half, access::read> keep [[texture(1)]], texture2d<float, access::write> out [[texture(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    if (g.x >= out.get_width() || g.y >= out.get_height()) return;
    if (keep.read(min(g / 4, uint2(keep.get_width() - 1, keep.get_height() - 1))).r > 0.5h) out.write(cur.read(g), g);
}
)FGMV";

enum { FGMV_LUMA, FGMV_SEARCH, FGMV_PICK, FGMV_STILL, FGMV_HUD, FGMV_KEEP, FGMV_COUNT };

/// What frame generation needs for frames of one format and size.
struct FrameGenSet {
    NSUInteger format{}, width{}, height{};
    id fi; ///< id<MTLFXFrameInterpolator> (iOS/macOS 26); nil: MetalFX makes none for these
    id<MTLTexture> history[2], luma[2]; ///< the last two presented frames and their 1/8 luma
    id<MTLTexture> gen, depth, motion, tiles, still, keep;
    int cur{};    ///< the slot the next copy goes to
    int filled{}; ///< presents copied since the history started (or restarted)
    bool clear{}, keepable{};
    uint64_t used{};
};

struct FrameGen {
    bool checked{}, on{}, mv{};
    id<MTLComputePipelineState> ps[FGMV_COUNT];
    FrameGenSet sets[2]; ///< the game's frame and another size (menus, movies) without remaking
    uint64_t clock{};
    double last_present{};      ///< seconds, the previous present
    uint64_t presents{}, paused_until{};
    uint64_t drops[3]{};        ///< the present numbers of the last three late presents
    uint64_t made{}, late{}, paused{};
};

FrameGen& GetFrameGen() {
    static FrameGen g;
    return g;
}

double NowSeconds() {
    return double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) * 1e-9;
}

bool FrameGenEnabled(id<MTLDevice> device) {
    auto& g = GetFrameGen();
    if (g.checked) {
        return g.on;
    }
    g.checked = true;
    const char* env = std::getenv("BB_FRAMEGEN");
    if (!env || env[0] != '1') {
        return false;
    }
    if (@available(iOS 26.0, macOS 26.0, *)) {
        dlopen("/System/Library/Frameworks/MetalFX.framework/MetalFX", RTLD_LAZY | RTLD_LOCAL);
        Class cls = NSClassFromString(@"MTLFXFrameInterpolatorDescriptor");
        if (cls && [cls respondsToSelector:@selector(supportsDevice:)] &&
            [cls supportsDevice:device]) {
            g.on = true;
            std::printf("Frame generation (BB_FRAMEGEN=1): MetalFX's frame interpolator makes a "
                        "frame between each two presents, shown a 60 Hz period before the game's\n");
            const char* mv = std::getenv("BB_FRAMEGEN_MV");
            if (!(mv && mv[0] == '0')) {
                NSError* error = nil;
                id<MTLLibrary> lib = [device newLibraryWithSource:@(FgmvSource) options:nil
                                                            error:&error];
                static const char* names[FGMV_COUNT] = {"fgmv_luma", "fgmv_search", "fgmv_pick",
                                                        "fgmv_still", "fgmv_hud", "fgmv_keep"};
                g.mv = lib != nil;
                for (int i = 0; g.mv && i < FGMV_COUNT; ++i) {
                    id<MTLFunction> f = [lib newFunctionWithName:@(names[i])];
                    g.ps[i] = f ? [device newComputePipelineStateWithFunction:f error:&error] : nil;
                    g.mv = g.ps[i] != nil;
                }
                if (g.mv && g.ps[FGMV_SEARCH].maxTotalThreadsPerThreadgroup < 256) {
                    g.mv = false;
                }
                std::printf(g.mv ? "Frame generation: motion found from the two pictures, the HUD "
                                   "kept (BB_FRAMEGEN_MV=0: off)\n"
                                 : "Frame generation: no motion search (%s): MetalFX's own flow\n",
                            error ? error.localizedDescription.UTF8String : "the GPU");
            }
            return true;
        }
    }
    std::printf("Frame generation: this device has no MetalFX frame interpolator; frames are "
                "shown as the game draws them\n");
    return false;
}

id<MTLTexture> NewTexture(id<MTLDevice> device, MTLPixelFormat format, NSUInteger width,
                          NSUInteger height, MTLTextureUsage usage) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                 width:width
                                                                                height:height
                                                                             mipmapped:NO];
    d.storageMode = MTLStorageModePrivate;
    d.usage = usage;
    return [device newTextureWithDescriptor:d];
}

bool KeepableFormat(MTLPixelFormat f) {
    return f == MTLPixelFormatRGBA8Unorm || f == MTLPixelFormatBGRA8Unorm ||
           f == MTLPixelFormatRGBA16Float || f == MTLPixelFormatRGB10A2Unorm ||
           f == MTLPixelFormatRG11B10Float || f == MTLPixelFormatRGBA32Float;
}

/// The set for frames of this format and size (made when new; two kept, the least recently used
/// replaced); nullptr when MetalFX makes no interpolator for them.
API_AVAILABLE(ios(26.0), macos(26.0))
FrameGenSet* FrameGenFor(FrameGen& g, id<MTLDevice> device, id<MTLTexture> frame) {
    ++g.clock;
    FrameGenSet* lru = &g.sets[0];
    for (auto& set : g.sets) {
        if (set.width && set.format == frame.pixelFormat && set.width == frame.width &&
            set.height == frame.height) {
            set.used = g.clock;
            return set.fi ? &set : nullptr;
        }
        if (set.used < lru->used) {
            lru = &set;
        }
    }
    FrameGenSet& s = *lru;
    s = FrameGenSet{};
    s.format = frame.pixelFormat;
    s.width = frame.width;
    s.height = frame.height;
    s.used = g.clock;
    // The depth and motion inputs at 1/8 of the picture's size when the motion is searched (its
    // vectors in those pixels), else at half size.
    const NSUInteger iw = g.mv ? std::max<NSUInteger>((frame.width + 7) / 8, 16)
                               : std::max<NSUInteger>(frame.width / 2, 16);
    const NSUInteger ih = g.mv ? std::max<NSUInteger>((frame.height + 7) / 8, 16)
                               : std::max<NSUInteger>(frame.height / 2, 16);
    Class cls = NSClassFromString(@"MTLFXFrameInterpolatorDescriptor");
    MTLFXFrameInterpolatorDescriptor* d = [[cls alloc] init];
    d.colorTextureFormat = frame.pixelFormat;
    d.outputTextureFormat = frame.pixelFormat;
    d.depthTextureFormat = MTLPixelFormatDepth32Float;
    d.motionTextureFormat = MTLPixelFormatRG16Float;
    d.inputWidth = iw;
    d.inputHeight = ih;
    d.outputWidth = frame.width;
    d.outputHeight = frame.height;
    id<MTLFXFrameInterpolator> fi = [d newFrameInterpolatorWithDevice:device];
    if (!fi) {
        std::printf("Frame generation: MetalFX makes no frame interpolator for %lux%lu format %lu; "
                    "frames of this size are shown as the game draws them\n",
                    (unsigned long)frame.width, (unsigned long)frame.height,
                    (unsigned long)frame.pixelFormat);
        return nullptr;
    }
    s.keepable = g.mv && KeepableFormat(frame.pixelFormat);
    const MTLTextureUsage color = MTLTextureUsageShaderRead | fi.colorTextureUsage;
    const MTLTextureUsage rw = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    for (int i = 0; i < 2; ++i) {
        s.history[i] = NewTexture(device, frame.pixelFormat, frame.width, frame.height, color);
        if (g.mv) {
            s.luma[i] = NewTexture(device, MTLPixelFormatR16Float, iw, ih, rw);
        }
    }
    s.gen = NewTexture(device, frame.pixelFormat, frame.width, frame.height,
                       fi.outputTextureUsage | MTLTextureUsageShaderRead |
                           (s.keepable ? MTLTextureUsageShaderWrite : 0));
    s.depth = NewTexture(device, MTLPixelFormatDepth32Float, iw, ih,
                         fi.depthTextureUsage | MTLTextureUsageRenderTarget);
    s.motion = NewTexture(device, MTLPixelFormatRG16Float, iw, ih,
                          fi.motionTextureUsage | MTLTextureUsageRenderTarget |
                              (g.mv ? MTLTextureUsageShaderWrite : 0));
    bool ok = s.history[0] && s.history[1] && s.gen && s.depth && s.motion;
    if (g.mv) {
        s.tiles = NewTexture(device, MTLPixelFormatRGBA16Float, (iw + 7) / 8, (ih + 7) / 8, rw);
        s.still = NewTexture(device, MTLPixelFormatR16Float, (frame.width + 3) / 4,
                             (frame.height + 3) / 4, rw);
        s.keep = NewTexture(device, MTLPixelFormatR16Float, (frame.width + 3) / 4,
                            (frame.height + 3) / 4, rw);
        ok = ok && s.luma[0] && s.luma[1] && s.tiles && s.still && s.keep;
    }
    if (!ok) {
        std::printf("Frame generation: no textures for %lux%lu\n", (unsigned long)frame.width,
                    (unsigned long)frame.height);
        return nullptr;
    }
    s.fi = fi;
    s.clear = true;
    std::printf("Frame generation: MetalFX frame interpolator %lux%lu format %lu (inputs %lux%lu%s)\n",
                (unsigned long)frame.width, (unsigned long)frame.height,
                (unsigned long)frame.pixelFormat, (unsigned long)iw, (unsigned long)ih,
                g.mv ? (s.keepable ? ", motion found, HUD kept" : ", motion found") : "");
    return &s;
}

void Dispatch(id<MTLComputeCommandEncoder> ce, id<MTLComputePipelineState> ps, NSUInteger w,
              NSUInteger h) {
    [ce setComputePipelineState:ps];
    [ce dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
}

/// Copies `frame` into the history and, when the frame before it is there, encodes the frame
/// between the two. Returns the texture to show first, or nil.
id<MTLTexture> FrameGenEncode(FrameGen& g, id<MTLDevice> device, id<MTLCommandBuffer> cmd,
                              id<MTLTexture> frame) {
    if (@available(iOS 26.0, macOS 26.0, *)) {
        // Late frames: a present more than 1.75 frames (at 30 FPS, 58 ms) after the one before
        // shows no generated frame; one a single 60 Hz refresh late (50 ms, common in busy
        // scenes) still does. Three late presents within 30 stop the interpolation for 30
        // presents (one second). The fork's stricter rule (any 50 ms frame, pauses doubling up to
        // 300) kept it paused for most of a Bloodborne session: 9,682 of about 13,000 presents.
        const double now = NowSeconds();
        const bool late = g.last_present > 0 && now - g.last_present > 1.75 / 30.0;
        g.last_present = now;
        ++g.presents;
        if (late) {
            ++g.late;
            g.drops[0] = g.drops[1];
            g.drops[1] = g.drops[2];
            g.drops[2] = g.presents;
            if (g.drops[0] && g.presents - g.drops[0] <= 30) {
                g.paused_until = g.presents + 30;
            }
        }
        FrameGenSet* sp = FrameGenFor(g, device, frame);
        if (!sp) {
            return nil;
        }
        FrameGenSet& s = *sp;
        if (s.clear) { // the constant inputs: far depth (0, reversed) and no motion
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = s.motion;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rp.depthAttachment.texture = s.depth;
            rp.depthAttachment.loadAction = MTLLoadActionClear;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            rp.depthAttachment.clearDepth = 0.0;
            [[cmd renderCommandEncoderWithDescriptor:rp] endEncoding];
            s.clear = false;
        }
        const int k = s.cur, prev = 1 - s.cur;
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromTexture:frame toTexture:s.history[k]];
        [blit endEncoding];
        s.cur = prev;
        if (g.mv) { // every present's luma, so the next one can search against it
            id<MTLComputeCommandEncoder> ce = [cmd computeCommandEncoder];
            [ce setTexture:s.history[k] atIndex:0];
            [ce setTexture:s.luma[k] atIndex:1];
            Dispatch(ce, g.ps[FGMV_LUMA], s.luma[k].width, s.luma[k].height);
            [ce endEncoding];
        }
        if (++s.filled < 2) {
            return nil; // nothing before it yet: the next present interpolates from this one
        }
        if (g.presents < g.paused_until) {
            ++g.paused;
            s.filled = 1; // the history starts again from this present when it resumes
            return nil;
        }
        if (g.mv) {
            id<MTLComputeCommandEncoder> ce = [cmd computeCommandEncoder];
            [ce setTexture:s.luma[k] atIndex:0];
            [ce setTexture:s.luma[prev] atIndex:1];
            [ce setTexture:s.tiles atIndex:2];
            [ce setComputePipelineState:g.ps[FGMV_SEARCH]];
            [ce dispatchThreadgroups:MTLSizeMake(s.tiles.width, s.tiles.height, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [ce setTexture:s.motion atIndex:3];
            Dispatch(ce, g.ps[FGMV_PICK], s.motion.width, s.motion.height);
            [ce endEncoding];
        }
        id<MTLFXFrameInterpolator> fi = (id<MTLFXFrameInterpolator>)s.fi;
        fi.colorTexture = s.history[k];
        fi.prevColorTexture = s.history[prev];
        fi.outputTexture = s.gen;
        fi.depthTexture = s.depth;
        fi.motionTexture = s.motion;
        fi.motionVectorScaleX = 1;
        fi.motionVectorScaleY = 1;
        fi.deltaTime = 1.0f / 30.0f;
        fi.nearPlane = 0.1f;
        fi.farPlane = 1000.0f;
        fi.fieldOfView = 60.0f;
        fi.aspectRatio = float(frame.width) / float(frame.height);
        fi.jitterOffsetX = 0;
        fi.jitterOffsetY = 0;
        fi.depthReversed = YES;
        fi.shouldResetHistory = s.filled == 2;
        [fi encodeToCommandBuffer:cmd];
        if (s.keepable) { // the still HUD from the game's frame
            id<MTLComputeCommandEncoder> ce = [cmd computeCommandEncoder];
            [ce setTexture:s.history[k] atIndex:0];
            [ce setTexture:s.history[prev] atIndex:1];
            [ce setTexture:s.still atIndex:2];
            Dispatch(ce, g.ps[FGMV_STILL], s.still.width, s.still.height);
            [ce setTexture:s.still atIndex:0];
            [ce setTexture:s.keep atIndex:1];
            Dispatch(ce, g.ps[FGMV_HUD], s.keep.width, s.keep.height);
            [ce setTexture:s.history[k] atIndex:0];
            [ce setTexture:s.keep atIndex:1];
            [ce setTexture:s.gen atIndex:2];
            Dispatch(ce, g.ps[FGMV_KEEP], s.gen.width, s.gen.height);
            [ce endEncoding];
        }
        // The first two after a start are the new frame unchanged (the fork's probe): shown from
        // the third on. A late present shows its own frame only.
        if (s.filled < 4 || late) {
            return nil;
        }
        if (!(++g.made % 1800)) {
            std::printf("Frame generation: %llu frames generated; %llu late presents, %llu "
                        "presents paused\n",
                        (unsigned long long)g.made, (unsigned long long)g.late,
                        (unsigned long long)g.paused);
        }
        return s.gen;
    }
    return nil;
}

} // namespace

bool Present(void* layer_ptr, PresentInfo info) {
    @autoreleasepool {
        CAMetalLayer* layer = (__bridge CAMetalLayer*)layer_ptr;
        id<MTLDevice> device = layer.device;
        if (FrameGenEnabled(device)) {
            // The generated frame first, in a drawable of its own: it is shown at the next
            // refresh and the game's frame (below) one 60 Hz period after it.
            id<MTLTexture> source = MetalFrame::ImageTexture(static_cast<VkImage>(info.frame));
            auto& g = GetFrameGen();
            id<MTLCommandQueue> fq = (__bridge id<MTLCommandQueue>)MetalLayer::MtlQueue();
            id<MTLCommandBuffer> fcmd = source ? [fq commandBuffer] : nil;
            if (fcmd) {
                fcmd.label = @"bbport frame generation";
                id<MTLTexture> generated = FrameGenEncode(g, device, fcmd, source);
                if (generated && info.fit.width && info.fit.height) {
                    auto& fpasses = GetPasses(device);
                    id<MTLRenderPipelineState> fpipe = Pipeline(fpasses, device, layer.pixelFormat);
                    id<CAMetalDrawable> fdrawable = fpipe ? [layer nextDrawable] : nil;
                    if (fdrawable) {
                        MTLRenderPassDescriptor* fpass = [MTLRenderPassDescriptor renderPassDescriptor];
                        fpass.colorAttachments[0].texture = fdrawable.texture;
                        fpass.colorAttachments[0].loadAction = MTLLoadActionClear;
                        fpass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
                        fpass.colorAttachments[0].storeAction = MTLStoreActionStore;
                        id<MTLRenderCommandEncoder> fenc = [fcmd renderCommandEncoderWithDescriptor:fpass];
                        fenc.label = @"bbport generated frame";
                        [fenc setRenderPipelineState:fpipe];
                        [fenc setViewport:(MTLViewport){double(info.fit.x), double(info.fit.y),
                                                        double(info.fit.width), double(info.fit.height),
                                                        0.0, 1.0}];
                        [fenc setFragmentTexture:generated atIndex:0];
                        [fenc setFragmentSamplerState:fpasses.sampler atIndex:0];
                        [fenc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [fenc endEncoding];
#if TARGET_OS_IPHONE
                        [fcmd presentDrawable:fdrawable afterMinimumDuration:1.0 / 60.0];
#else
                        [fcmd presentDrawable:fdrawable];
#endif
                    }
                }
                [fcmd commit];
            }
        }
        id<CAMetalDrawable> drawable = [layer nextDrawable];
        id<MTLTexture> frame = MetalFrame::ImageTexture(static_cast<VkImage>(info.frame));
        auto& passes = GetPasses(device);
        id<MTLRenderPipelineState> pipeline =
            drawable ? Pipeline(passes, device, drawable.texture.pixelFormat) : nil;
        if (!drawable || !frame || !pipeline) {
            if (info.done) {
                info.done();
            }
            return false;
        }
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)MetalLayer::MtlQueue();
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        cmd.label = @"bbport Metal present";
        id<MTLTexture> target = drawable.texture;

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear; // the letterbox in black
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> encoder = [cmd renderCommandEncoderWithDescriptor:pass];
        encoder.label = @"bbport present";
        if (info.fit.width && info.fit.height) {
            [encoder setRenderPipelineState:pipeline];
            [encoder setViewport:(MTLViewport){double(info.fit.x), double(info.fit.y),
                                               double(info.fit.width), double(info.fit.height),
                                               0.0, 1.0}];
            [encoder setFragmentTexture:frame atIndex:0];
            [encoder setFragmentSamplerState:passes.sampler atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
        [encoder endEncoding];
        if (info.overlay) {
            info.overlay((__bridge void*)cmd, (__bridge void*)target, u32(target.width),
                         u32(target.height));
        }

        id<MTLBuffer> dump;
        if (!info.dump_path.empty()) {
            const NSUInteger w = target.width, h = target.height;
            dump = [device newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            [blit copyFromTexture:target
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:dump
                   destinationOffset:0
              destinationBytesPerRow:w * 4
            destinationBytesPerImage:w * h * 4];
            [blit endEncoding];
        }
#if TARGET_OS_IPHONE
        // ProMotion (iPad Pro, 120 Hz): a present with no cadence shows at the next 120 Hz
        // refresh and keeps the panel at 120 Hz for 60 FPS content. A minimum duration of one
        // 60 Hz period tells the system the game's rate, so the display can refresh at 60 Hz
        // (less power, steadier pacing). The game never presents faster than its 60 Hz vblank.
        // With a frame limit that divides 120 (BB_FPS_LIMIT=40 or 30) and no frame generation,
        // each frame is held for 1/limit s: on a 120 Hz screen 40 FPS is 3 refreshes a frame,
        // evenly, where a 60 Hz cadence alternated 1 and 2 refreshes.
        static const double period = [] {
            const char* env = std::getenv("BB_FPS_LIMIT");
            const long limit = env ? std::strtol(env, nullptr, 10) : 0;
            return limit > 0 && limit < 60 && 120 % limit == 0 ? 1.0 / double(limit) : 1.0 / 60.0;
        }();
        [cmd presentDrawable:drawable
            afterMinimumDuration:FrameGenEnabled(device) ? 1.0 / 60.0 : period];
#else
        [cmd presentDrawable:drawable];
#endif
        __block std::function<void()> on_done = std::move(info.done);
        __block std::string path = std::move(info.dump_path);
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
          if (dump) {
              if (FILE* f = std::fopen(path.c_str(), "wb")) {
                  std::fwrite(dump.contents, 1, dump.length, f);
                  std::fclose(f);
              }
          }
          if (on_done) {
              on_done();
          }
        }];
        [cmd commit];
        return true;
    }
}

} // namespace MetalPresent
