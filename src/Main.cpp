#include <iostream>
#include <Strawberry/Window/Window.hpp>
#include <Strawberry/Vulkan/Surface.hpp>
#include <Strawberry/Vulkan/Device.hpp>
#include <Strawberry/Vulkan/Instance.hpp>
#include <Strawberry/Vulkan/Swapchain.hpp>
#include <Strawberry/Vulkan/CommandBuffer.hpp>
#include <Strawberry/Vulkan/CommandPool.hpp>
#include "Game/Board.hpp"
#include "Strawberry/Vulkan/Framebuffer.hpp"
#include "Strawberry/Vulkan/RenderPass.hpp"


int main()
{
	using namespace Strawberry;
	using namespace RLPuyo;

	Strawberry::RLPuyo::Board board;


	Window::Window window("RLPuyo", {480, 360});
	Vulkan::Instance instance;
	const Vulkan::PhysicalDevice& physicalDevice = instance.GetPhysicalDevices()[0];


	const unsigned int queueFamily = physicalDevice.SearchQueueFamilies(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)[0];
	Vulkan::Device device(physicalDevice, {
							  Vulkan::QueueCreateInfo{
								  .familyIndex = queueFamily,
								  .count = 1,
							  }});
	Vulkan::Surface surface(window, device);

	Core::ReflexivePointer<Vulkan::Queue> queue = device.GetQueue(queueFamily, 0);
	Vulkan::Swapchain swapchain(*queue, surface, window.GetSize());

	Vulkan::CommandPool commandPool(*queue);
	Vulkan::CommandBuffer commandBuffer(commandPool);
	Vulkan::RenderPass renderPass = Vulkan::RenderPass::Builder(device)
		.WithColorAttachment(VK_FORMAT_R32G32B32A32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
		.WithSubpass(Vulkan::SubpassDescription().WithColorAttachment(0))
		.Build();
	Vulkan::Framebuffer framebuffer(renderPass, window.GetSize().AsType<unsigned int>());

	while (!window.CloseRequested())
	{
		Window::PollInput();

		queue->WaitUntilIdle();
		commandBuffer.Reset();

		auto* swapchainImage = swapchain.GetNextImage().Unwrap();
		commandBuffer.Begin(true);
		commandBuffer.BeginRenderPass(renderPass, framebuffer);
		commandBuffer.EndRenderPass();
		commandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_UNDEFINED)
				.ToLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		});
		commandBuffer.BlitImage(framebuffer.GetColorAttachment(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *swapchainImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT, VK_FILTER_LINEAR);
		commandBuffer.PipelineBarrier(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, {
			Vulkan::ImageMemoryBarrier(*swapchainImage, VK_IMAGE_ASPECT_COLOR_BIT)
				.FromLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
				.ToLayout(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
		});
		commandBuffer.End();
		queue->Submit(commandBuffer);
		swapchain.Present();
		window.SwapBuffers();
	}

	queue->WaitUntilIdle();
	return 0;
}
