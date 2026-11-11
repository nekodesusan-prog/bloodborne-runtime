// SPDX-License-Identifier: GPL-2.0-or-later
// The render adapter's Metal presents (mtl_present.h). ARC.

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dlfcn.h>

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
/// at 60). Adapted from cheemu21/kid-with-sword-remake (nxnative fork, MIT, commit a758ea2,
/// dm/gfx/nvn/nvn_mtl.m fg_for/fg_encode): the interpolator works from the finished pictures
/// alone, with a constant far depth (reversed, 0) and no motion, and MetalFX finds the motion.
struct FrameGen {
    bool checked{}, on{}, bad{};
    id fi; ///< id<MTLFXFrameInterpolator> (iOS/macOS 26)
    NSUInteger format{}, width{}, height{};
    id<MTLTexture> history[2]; ///< copies of the last two presented frames
    id<MTLTexture> gen, depth, motion;
    int cur{};                 ///< the slot the next copy goes to
    int filled{};              ///< presents copied since the history started (or restarted)
    bool clear{};
    uint64_t made{}, resets{};
};

FrameGen& GetFrameGen() {
    static FrameGen g;
    return g;
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

/// The interpolator and its textures for frames of this format and size (made again when either
/// changes); false when MetalFX makes none.
API_AVAILABLE(ios(26.0), macos(26.0))
bool FrameGenFor(FrameGen& g, id<MTLDevice> device, id<MTLTexture> frame) {
    if ((g.fi || g.bad) && g.format == frame.pixelFormat && g.width == frame.width &&
        g.height == frame.height) {
        return g.fi != nil;
    }
    g.fi = nil;
    g.bad = true; // until everything below is made
    g.format = frame.pixelFormat;
    g.width = frame.width;
    g.height = frame.height;
    g.filled = 0;
    Class cls = NSClassFromString(@"MTLFXFrameInterpolatorDescriptor");
    MTLFXFrameInterpolatorDescriptor* d = [[cls alloc] init];
    d.colorTextureFormat = frame.pixelFormat;
    d.outputTextureFormat = frame.pixelFormat;
    d.depthTextureFormat = MTLPixelFormatDepth32Float;
    d.motionTextureFormat = MTLPixelFormatRG16Float;
    d.inputWidth = frame.width;
    d.inputHeight = frame.height;
    d.outputWidth = frame.width;
    d.outputHeight = frame.height;
    id<MTLFXFrameInterpolator> fi = [d newFrameInterpolatorWithDevice:device];
    if (!fi) {
        std::printf("Frame generation: MetalFX makes no frame interpolator for %lux%lu format %lu; "
                    "frames are shown as the game draws them\n",
                    (unsigned long)frame.width, (unsigned long)frame.height,
                    (unsigned long)frame.pixelFormat);
        return false;
    }
    const MTLTextureUsage color = MTLTextureUsageShaderRead | fi.colorTextureUsage;
    g.history[0] = NewTexture(device, frame.pixelFormat, frame.width, frame.height, color);
    g.history[1] = NewTexture(device, frame.pixelFormat, frame.width, frame.height, color);
    g.gen = NewTexture(device, frame.pixelFormat, frame.width, frame.height,
                       fi.outputTextureUsage | MTLTextureUsageShaderRead);
    g.depth = NewTexture(device, MTLPixelFormatDepth32Float, frame.width, frame.height,
                         fi.depthTextureUsage | MTLTextureUsageRenderTarget);
    g.motion = NewTexture(device, MTLPixelFormatRG16Float, frame.width, frame.height,
                          fi.motionTextureUsage | MTLTextureUsageRenderTarget);
    if (!g.history[0] || !g.history[1] || !g.gen || !g.depth || !g.motion) {
        std::printf("Frame generation: no textures for %lux%lu\n", (unsigned long)frame.width,
                    (unsigned long)frame.height);
        return false;
    }
    g.fi = fi;
    g.bad = false;
    g.clear = true;
    std::printf("Frame generation: MetalFX frame interpolator %lux%lu format %lu\n",
                (unsigned long)frame.width, (unsigned long)frame.height,
                (unsigned long)frame.pixelFormat);
    return true;
}

/// Copies `frame` into the history and, when the frame before it is there, encodes the frame
/// between the two into g.gen. True when g.gen holds a frame to show.
bool FrameGenEncode(FrameGen& g, id<MTLDevice> device, id<MTLCommandBuffer> cmd,
                    id<MTLTexture> frame) {
    if (@available(iOS 26.0, macOS 26.0, *)) {
        if (!FrameGenFor(g, device, frame)) {
            return false;
        }
        if (g.clear) { // the constant inputs: far depth (0, reversed) and no motion
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = g.motion;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            rp.depthAttachment.texture = g.depth;
            rp.depthAttachment.loadAction = MTLLoadActionClear;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            rp.depthAttachment.clearDepth = 0.0;
            [[cmd renderCommandEncoderWithDescriptor:rp] endEncoding];
            g.clear = false;
        }
        const int k = g.cur, prev = 1 - g.cur;
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromTexture:frame toTexture:g.history[k]];
        [blit endEncoding];
        g.cur = prev;
        if (++g.filled < 2) {
            return false; // nothing before it yet: the next present interpolates from this one
        }
        id<MTLFXFrameInterpolator> fi = (id<MTLFXFrameInterpolator>)g.fi;
        fi.colorTexture = g.history[k];
        fi.prevColorTexture = g.history[prev];
        fi.outputTexture = g.gen;
        fi.depthTexture = g.depth;
        fi.motionTexture = g.motion;
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
        fi.shouldResetHistory = g.filled == 2;
        [fi encodeToCommandBuffer:cmd];
        // The first two after a start are the new frame unchanged (the fork's probe): shown from
        // the third on.
        if (g.filled < 4) {
            return false;
        }
        if (!(++g.made % 1800)) {
            std::printf("Frame generation: %llu frames generated\n", (unsigned long long)g.made);
        }
        return true;
    }
    return false;
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
                if (FrameGenEncode(g, device, fcmd, source) && info.fit.width && info.fit.height) {
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
                        [fenc setFragmentTexture:g.gen atIndex:0];
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
        [cmd presentDrawable:drawable afterMinimumDuration:1.0 / 60.0];
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
