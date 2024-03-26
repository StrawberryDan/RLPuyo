#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
// Strawberry Vulkan
#include "Strawberry/Vulkan/Framebuffer.hpp"
#include "Strawberry/Vulkan/RenderPass.hpp"
#include <Strawberry/Vulkan/CommandBuffer.hpp>
#include <Strawberry/Vulkan/CommandPool.hpp>
#include <Strawberry/Vulkan/Device.hpp>
#include <Strawberry/Vulkan/Instance.hpp>
#include <Strawberry/Vulkan/Surface.hpp>
#include <Strawberry/Vulkan/Swapchain.hpp>


//======================================================================================================================
//  Foreward Declarations
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::Window { class Window; }


//======================================================================================================================
//  Class Declaration
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
    class Renderer
    {
    public:
        Renderer(Window::Window& window);
        ~Renderer();


        void Render();

   
    private:
        Vulkan::Instance mInstance;
        const Vulkan::PhysicalDevice* mPhysicalDevice;
        const unsigned int mQueueFamily;
        Vulkan::Device mDevice;
        Core::ReflexivePointer<Vulkan::Queue> mQueue;
        Vulkan::Surface mSurface;
        Vulkan::Swapchain mSwapchain;
        Vulkan::RenderPass mRenderPass;
        Vulkan::Framebuffer mFramebuffer;
        Vulkan::CommandPool mCommandPool;
        Vulkan::CommandBuffer mCommandBuffer;
    };
}