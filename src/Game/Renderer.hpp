#pragma once
//======================================================================================================================
//  Includes
//----------------------------------------------------------------------------------------------------------------------
// RL PUYO
#include "Board.hpp"
// Strawberry Vulkan
#include "Strawberry/Vulkan/CommandBuffer.hpp"
#include "Strawberry/Vulkan/CommandPool.hpp"
#include "Strawberry/Vulkan/Device.hpp"
#include "Strawberry/Vulkan/Framebuffer.hpp"
#include "Strawberry/Vulkan/GraphicsPipeline.hpp"
#include "Strawberry/Vulkan/Instance.hpp"
#include "Strawberry/Vulkan/NaiveAllocator.hpp"
#include "Strawberry/Vulkan/RenderPass.hpp"
#include "Strawberry/Vulkan/Surface.hpp"
#include "Strawberry/Vulkan/Swapchain.hpp"


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


		void Submit(const Board& board);


		void Render();


	private:
		static Vulkan::PipelineLayout   CreatePipelineLayout(const Vulkan::Device& device, const Vulkan::RenderPass& renderPass);
		static Vulkan::GraphicsPipeline CreatePipeline(Core::Math::Vec2f renderSize, const Vulkan::Device& device,
													   const Vulkan::RenderPass& renderPass,
													   const Vulkan::PipelineLayout& layout);


	private:
		Core::Math::Vec2f                     mRenderSize;


		Vulkan::Instance                      mInstance;
		const Vulkan::PhysicalDevice*         mPhysicalDevice;
		const unsigned int                    mQueueFamily;
		Vulkan::Device                        mDevice;
		Vulkan::NaiveAllocator                mAllocator;
		Core::ReflexivePointer<Vulkan::Queue> mQueue;
		Vulkan::Surface                       mSurface;
		Vulkan::Swapchain                     mSwapchain;
		Vulkan::RenderPass                    mRenderPass;
		Vulkan::Framebuffer                   mFramebuffer;
		Vulkan::PipelineLayout                mPipelineLayout;
		Vulkan::GraphicsPipeline              mPipeline;
		Vulkan::DescriptorPool                mDescriptorPool;
		Vulkan::DescriptorSet                 mConstantsDescriptor;
		Vulkan::Buffer                        mConstantsBuffer;
		Vulkan::DescriptorSet                 mBoardStateDescriptor;
		Vulkan::Buffer                        mBoardStateBuffer;
		Vulkan::CommandPool                   mCommandPool;
		Vulkan::CommandBuffer                 mPrimaryCommandBuffer;
		Vulkan::CommandBuffer                 mDrawingBuffer;
	};
}