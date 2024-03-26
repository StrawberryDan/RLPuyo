//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
#include "Renderer.hpp"

#include "Strawberry/Window/Window.hpp"


//======================================================================================================================
//  Method Definitions
//----------------------------------------------------------------------------------------------------------------------
namespace Strawberry::RLPuyo
{
	Renderer::Renderer(Window::Window& window)
		: mInstance()
		, mPhysicalDevice(&mInstance.GetPhysicalDevices()[0])
		, mQueueFamily(mPhysicalDevice->SearchQueueFamilies(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)[0])
		, mDevice(*mPhysicalDevice, {Vulkan::QueueCreateInfo(mQueueFamily, 1)})
		, mQueue(mDevice.GetQueue(mQueueFamily, 0))
		, mSurface(window, mDevice)
		, mSwapchain(*mQueue, mSurface, window.GetSize())
		, mRenderPass(Vulkan::RenderPass::Builder(mDevice)
			.WithColorAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
			.WithSubpass(Vulkan::SubpassDescription().WithColorAttachment(0))
			.Build())
		, mFramebuffer(mRenderPass, window.GetSize().AsType<unsigned int>())
		, mCommandPool(*mQueue)
		, mCommandBuffer(mCommandPool)
	{}


	Renderer::~Renderer()
	{
		mQueue->WaitUntilIdle();
	}


	void Renderer::Render()
	{
		mQueue->WaitUntilIdle();
		mCommandBuffer.Reset();

		auto* swapchainImage = mSwapchain.GetNextImage().Unwrap();
		mCommandBuffer.Begin(true);
		mCommandBuffer.BeginRenderPass(mRenderPass, mFramebuffer);
		mCommandBuffer.EndRenderPass();
		mCommandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_UNDEFINED)
				.ToLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		});
		mCommandBuffer.BlitImage(mFramebuffer.GetColorAttachment(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR);
		mCommandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
				.ToLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
		});
		mCommandBuffer.End();
		mQueue->Submit(mCommandBuffer);
		mSwapchain.Present();
	}
}
