// SPDX-License-Identifier: GPL-2.0-or-later
// The render adapter's Metal presents (mtl_present.h). ARC.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

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

} // namespace

bool Present(void* layer_ptr, PresentInfo info) {
    @autoreleasepool {
        CAMetalLayer* layer = (__bridge CAMetalLayer*)layer_ptr;
        id<MTLDevice> device = layer.device;
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
