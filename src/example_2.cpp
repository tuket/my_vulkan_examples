#include <stdio.h>
#include <assert.h>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <span>
#include <optional>
#include <vector>

#define SHADERS_FOLDER "shaders/"

typedef uint32_t u32;
typedef uint8_t u8;
typedef const char* CStr;

static constexpr size_t MAX_ATTACHMENTS = 8;
static const auto VK_COLOR_COMPONENT_RGBA_BITS =
	VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

struct SwapChain_doubleBuff {
	u32 w, h;
	VkSwapchainKHR swapchain = VK_NULL_HANDLE;
	VkImageView imageViews[2];
	VkSemaphore semaphore_swapchainImgAvailable[2];
	VkSemaphore semaphore_drawFinished[2];
};

struct Vk {
	GLFWwindow* window;
	VkInstance instance;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice;
	u32 queueFamily;
	VkDevice device;
	SwapChain_doubleBuff swapchainDb;
	VkFramebuffer framebuffers[2];
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkCommandPool cmdPool;
};
static Vk vk;

static bool loadBinaryFile(std::vector<u8>& buffer, CStr fileName)
{
	FILE* file = fopen(fileName, "rb");
	if (!file)
		return false;
	fseek(file, 0, SEEK_END);
	const auto len = ftell(file);
	fseek(file, 0, SEEK_SET);
	buffer.resize(len);
	fread(buffer.data(), 1, len, file);
	fclose(file);
	return true;
}

static void createVulkanInstance(u32 apiVersion,
	std::span<const CStr> layers,
	std::span<const CStr> extensions,
	CStr appName, u32 appVersion = VK_MAKE_VERSION(1, 0, 0),
	CStr engineName = nullptr, u32 engineVersion = VK_MAKE_VERSION(1, 0, 0))
{
	const VkApplicationInfo appInfo = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = appName,
		.applicationVersion = appVersion,
		.pEngineName = engineName,
		.engineVersion = engineVersion,
		.apiVersion = apiVersion,
	};
	const VkInstanceCreateInfo instInfo = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &appInfo,
		.enabledLayerCount = u32(layers.size()),
		.ppEnabledLayerNames = layers.size() == 0 ? nullptr : &layers[0],
		.enabledExtensionCount = u32(extensions.size()),
		.ppEnabledExtensionNames = extensions.size() == 0 ? nullptr :&extensions[0],
	};
	VkResult vkRes = vkCreateInstance(&instInfo, nullptr, &vk.instance);
	assert(vkRes == VK_SUCCESS);
}

static VkPhysicalDevice findBestPhysicalDevice(VkInstance instance)
{
	constexpr u32 MAX_PHYSICAL_DEVICES = 8;
	VkPhysicalDevice physicalDevices[MAX_PHYSICAL_DEVICES];
	u32 numPhysicalDevices = std::size(physicalDevices);
	VkResult vkRes = vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, physicalDevices);
	assert(vkRes == VK_SUCCESS && numPhysicalDevices > 0);

	VkPhysicalDeviceProperties props[MAX_PHYSICAL_DEVICES];
	for (u32 i = 0; i < numPhysicalDevices; i++)
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props[i]);

	VkPhysicalDeviceMemoryProperties memProps[MAX_PHYSICAL_DEVICES];
	for (u32 i = 0; i < numPhysicalDevices; i++)
		vkGetPhysicalDeviceMemoryProperties(physicalDevices[i], &memProps[i]);

	auto physicalDeviceTypeScore = [](VkPhysicalDeviceType t) {
		switch (t) {
		case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
			return 0;
		case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
			return 1;
		case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
			return 2;
		case VK_PHYSICAL_DEVICE_TYPE_CPU:
			return 3;
		default:
			return 4;
		}
	};

	auto calcDeviceLocalMemory = [](const VkPhysicalDeviceMemoryProperties& memProps)
	{
		u32 total = 0;
		for (u32 i = 0; i < memProps.memoryHeapCount; i++) {
			auto& heap = memProps.memoryHeaps[i];
			if (heap.flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
				total += heap.size;
		}
		return total;
	};

	auto isBetter = [&](u32 a, u32 b) {
		const auto typeScoreA = physicalDeviceTypeScore(props[a].deviceType);
		const auto typeScoreB = physicalDeviceTypeScore(props[b].deviceType);
		if (typeScoreA != typeScoreB)
			return typeScoreA < typeScoreB;

		const u32 memA = calcDeviceLocalMemory(memProps[a]);
		const u32 memB = calcDeviceLocalMemory(memProps[b]);
		return memA > memB;
	};

	u32 bestI = 0;
	for (u32 i = 1; i < numPhysicalDevices; i++) {
		if (isBetter(i, bestI))
			bestI = i;
	}

	return physicalDevices[bestI];
}

static u32 findGraphicsQueueFamily(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
	constexpr u32 MAX_QUEUE_FAMILIES = 16;
	VkQueueFamilyProperties props[MAX_QUEUE_FAMILIES];
	u32 numQueueFamilies = 16;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilies, props);
	u32 queueFamily;
	for (queueFamily = 0; 0 < numQueueFamilies; queueFamily++) {
		const bool supportsGraphics = props[queueFamily].queueFlags & VK_QUEUE_GRAPHICS_BIT;
		VkBool32 supportsSurface;
		const VkResult vkRes = vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFamily, surface, &supportsSurface);
		assert(vkRes == VK_SUCCESS);
		if (supportsGraphics && supportsSurface)
			break;
	}
	assert(queueFamily < numQueueFamilies);
	return queueFamily;
}

struct CreateQueues {
	u32 family;
	std::span<const float> priorities;
};

[[nodiscard]]
static VkDevice createVulkanDevice(
	VkPhysicalDevice physicalDevice,
	std::span<const CreateQueues> queuesToCreate,
	std::span<const CStr> extensions = {}, const VkPhysicalDeviceFeatures& features = {})
{
	VkDeviceQueueCreateInfo queueCreateInfos[16];
	assert(queuesToCreate.size() < std::size(queueCreateInfos));
	for (size_t i = 0; i < queuesToCreate.size(); i++) {
		queueCreateInfos[i] = {
			.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.queueFamilyIndex = queuesToCreate[i].family,
			.queueCount = (u32)queuesToCreate[i].priorities.size(),
			.pQueuePriorities = &queuesToCreate[i].priorities[0],
		};
	}
	const VkDeviceCreateInfo deviceInfo = {
		.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
		.queueCreateInfoCount = u32(queuesToCreate.size()),
		.pQueueCreateInfos = queueCreateInfos,
		.enabledExtensionCount = u32(extensions.size()),
		.ppEnabledExtensionNames = extensions.size() == 0 ? nullptr : &extensions[0],
		.pEnabledFeatures = &features,
	};
	VkDevice device;
	const VkResult vkRes = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
	assert(vkRes == VK_SUCCESS);
	return device;
}

static void create_swapChain_doubleBuff(SwapChain_doubleBuff& o, VkPhysicalDevice physicalDevice, VkDevice device, VkSurfaceKHR surface)
{
	VkSurfaceCapabilitiesKHR surfaceCaps;
	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCaps);
	o.w = surfaceCaps.currentExtent.width;
	o.h = surfaceCaps.currentExtent.height;
	const VkSwapchainKHR oldSwapchain = o.swapchain;

	const VkSwapchainCreateInfoKHR swapchainInfo = {
		.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
		.surface = surface,
		.minImageCount = 2,
		.imageFormat = VK_FORMAT_B8G8R8A8_SRGB,
		.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, // the way the swapchain interprets image data
		.imageExtent = {o.w, o.h},
		.imageArrayLayers = 1,
		.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
		.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, // only one queue will access an image at the same time. We can still reference the image from different queues, but not at the same time. We must use a memory barrier for this!
		.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
		.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, // this can be used to have transparent windows in some window systems
		.presentMode = VK_PRESENT_MODE_MAILBOX_KHR, // with 2 images, MAILBOX and FIFO are equivalent
		.clipped = VK_TRUE, // this allows to discard the rendering of hidden pixel regions. E.g a window is partially covered by another
		.oldSwapchain = o.swapchain,
	};
	VkResult vkRes = vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &o.swapchain);
	assert(vkRes == VK_SUCCESS);

	// destroy old swapchain stuff
	if (oldSwapchain != VK_NULL_HANDLE) {
		for (int i = 0; i < 2; i++) {
			vkDestroySemaphore(device, o.semaphore_swapchainImgAvailable[i], nullptr);
			vkDestroySemaphore(device, o.semaphore_drawFinished[i], nullptr);
			
			vkDestroyImageView(device, o.imageViews[i], nullptr);
		}
	}

	// create image views
	VkImage images[2];
	u32 numImages = 2;
	vkRes = vkGetSwapchainImagesKHR(device, o.swapchain, &numImages, nullptr);
	assert(vkRes == VK_SUCCESS && numImages == 2);
	vkRes = vkGetSwapchainImagesKHR(device, o.swapchain, &numImages, images);
	assert(vkRes == VK_SUCCESS);
	for (u32 i = 0; i < 2; i++) {
		VkImageViewCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = images[i],
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = VK_FORMAT_B8G8R8A8_SRGB,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkRes = vkCreateImageView(device, &info, nullptr, o.imageViews + i);
		assert(vkRes == VK_SUCCESS);
	}

	// create semaphores
	const VkSemaphoreCreateInfo semInfo = { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	for (int i = 0; i < 2; i++) {
		vkRes = vkCreateSemaphore(device, &semInfo, nullptr, o.semaphore_swapchainImgAvailable + i);
		assert(vkRes == VK_SUCCESS);
		vkRes = vkCreateSemaphore(device, &semInfo, nullptr, o.semaphore_drawFinished + i);
		assert(vkRes == VK_SUCCESS);
	}
}

static VkRenderPass createSimpleRenderPass(
	std::span<const VkAttachmentDescription> attachments,
	bool lastIsDepthAttachment
)
{
	const u32 numAttachments = u32(attachments.size());
	assert(numAttachments > 0 && numAttachments <= MAX_ATTACHMENTS);
	VkAttachmentReference attachmentRefs[MAX_ATTACHMENTS];
	for (size_t i = 0; i < numAttachments; i++) {
		const bool isDepth = lastIsDepthAttachment && i == numAttachments - 1;
		attachmentRefs[i] = {
			.attachment = u32(i),
			.layout = isDepth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};
	}

	const u32 numColorAttachments = numAttachments - u32(lastIsDepthAttachment);
	const VkSubpassDescription subpassDesc = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.inputAttachmentCount = 0,
		.pInputAttachments = nullptr,
		.colorAttachmentCount = numColorAttachments,
		.pColorAttachments = attachmentRefs,
		.pDepthStencilAttachment = lastIsDepthAttachment ? &attachmentRefs[numColorAttachments] : nullptr,
	};

	VkRenderPass res;
	const VkRenderPassCreateInfo rpInfo = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = numAttachments,
		.pAttachments = &attachments[0],
		.subpassCount = 1,
		.pSubpasses = &subpassDesc,
		.dependencyCount = 0,
		.pDependencies = nullptr,
	};
	const VkResult vkRes = vkCreateRenderPass(vk.device, &rpInfo, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

static VkFramebuffer createFrameBuffer(VkDevice device, VkRenderPass renderPass,
	std::span<const VkImageView> imgViews, u32 width, u32 height)
{
	VkFramebuffer res;
	const VkFramebufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = renderPass,
		.attachmentCount = u32(imgViews.size()),
		.pAttachments = &imgViews[0],
		.width = width,
		.height = height,
		.layers = 1,
	};
	const VkResult vkRes = vkCreateFramebuffer(device, &info, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

static VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device, std::span<const VkDescriptorSetLayoutBinding> bindings)
{
	const VkDescriptorSetLayoutCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = u32(bindings.size()),
		.pBindings = &bindings[0],
	};
	VkDescriptorSetLayout res;
	const VkResult vkRes = vkCreateDescriptorSetLayout(device, &info, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

static VkPipelineLayout createPipelineLayout(VkDevice device,
	std::span<const VkDescriptorSetLayout> descriptorSetLayouts
	//std::span<const VkPushConstantRange> pushConstantRanges
	)
{
	const VkPipelineLayoutCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = u32(descriptorSetLayouts.size()),
		.pSetLayouts = &descriptorSetLayouts[0],
		.pushConstantRangeCount = 0,
	};
	VkPipelineLayout res;
	const VkResult vkRes = vkCreatePipelineLayout(device, &info, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

static VkPipeline createGraphicsPipeline(
	VkDevice device,
	std::span<const VkPipelineShaderStageCreateInfo> shaderStages,
	std::span<const VkVertexInputBindingDescription> vertexInputBindings,
	std::span<const VkVertexInputAttributeDescription> vertexInputAttribs,
	VkPrimitiveTopology primitiveTopology,
	u32 numAttachments, bool enableBlending,
	VkPipelineLayout layout,
	VkRenderPass renderPass, u32 subpass
)
{
	const VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = u32(vertexInputBindings.size()),
		.pVertexBindingDescriptions = &vertexInputBindings[0],
		.vertexAttributeDescriptionCount = u32(vertexInputAttribs.size()),
		.pVertexAttributeDescriptions = &vertexInputAttribs[0],
	};

	const VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = primitiveTopology,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineRasterizationStateCreateInfo rasterizationInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		//.depthBiasConstantFactor = 0,
		//.depthBiasClamp = VK_FALSE,
		//.depthBiasSlopeFactor = 0,
		//.lineWidth = 1.f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampleInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
	};

	assert(numAttachments <= MAX_ATTACHMENTS);
	VkPipelineColorBlendAttachmentState blendAttachments[MAX_ATTACHMENTS];
	for (u32 i = 0; i < numAttachments; i++) {
		blendAttachments[i] = {
			.blendEnable = enableBlending,
			.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.colorBlendOp = VK_BLEND_OP_ADD,
			.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
			.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
			.alphaBlendOp = VK_BLEND_OP_ADD,
			.colorWriteMask = VK_COLOR_COMPONENT_RGBA_BITS,
		};
	}
	const VkPipelineColorBlendStateCreateInfo blendInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = numAttachments,
		.pAttachments = blendAttachments,
	};

	const VkDynamicState dynamicStates[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const VkPipelineDynamicStateCreateInfo dynamicStateInfo = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = u32(std::size(dynamicStates)),
		.pDynamicStates = dynamicStates,
	};

	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = u32(shaderStages.size()),
		.pStages = &shaderStages[0],
		.pVertexInputState = &vertexInputInfo,
		.pInputAssemblyState = &inputAssemblyInfo,
		.pTessellationState = nullptr,
		.pViewportState = nullptr, // we will use dynamic stage for viewport and scissor
		.pRasterizationState = &rasterizationInfo,
		.pMultisampleState = &multisampleInfo,
		.pDepthStencilState = nullptr, // TODO
		.pColorBlendState = &blendInfo,
		.pDynamicState = &dynamicStateInfo,
		.layout = layout,
		.renderPass = renderPass,
		.subpass = 0,
		//.basePipelineHandle = VK_NULL_HANDLE,
		//.basePipelineIndex = 0,
	};

	VkPipeline res;
	const VkResult vkRes = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

static VkShaderModule loadSpivShaderModule(VkDevice device, CStr fileName)
{
	std::vector<u8> buffer;
	const bool ok = loadBinaryFile(buffer, fileName);
	assert(ok);
	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = buffer.size(),
		.pCode = (u32*)buffer.data(),
	};
	VkShaderModule res;
	vkCreateShaderModule(device, &info, nullptr, &res);
	return res;
}

static VkPipelineShaderStageCreateInfo makeShaderStageCreateInfo(VkShaderStageFlagBits stage, VkShaderModule module)
{
	return VkPipelineShaderStageCreateInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = stage,
		.module = module,
		.pName = "main",
		.pSpecializationInfo = nullptr, // allows to specify values for shader constants
	};
}

static VkCommandPool createCmdPool(VkDevice device, u32 queueFamily)
{
	const VkCommandPoolCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.queueFamilyIndex = queueFamily,
	};
	VkCommandPool res;
	const VkResult vkRes = vkCreateCommandPool(device, &info, nullptr, &res);
	assert(vkRes == VK_SUCCESS);
	return res;
}

void example_2()
{
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // glfw creates by default an OpenGL context, but this line avoids that
	vk.window = glfwCreateWindow(800, 600, "example 2", nullptr, nullptr);

	u32 numRequiredExtensions;
	const CStr* requiredExtensions = glfwGetRequiredInstanceExtensions(&numRequiredExtensions);

	createVulkanInstance(VK_API_VERSION_1_3, {}, {requiredExtensions, numRequiredExtensions}, "example 2");
	VkResult vkRes = glfwCreateWindowSurface(vk.instance, vk.window, nullptr, &vk.surface);
	assert(vkRes == VK_SUCCESS);
	vk.physicalDevice = findBestPhysicalDevice(vk.instance);
	vk.queueFamily = findGraphicsQueueFamily(vk.physicalDevice, vk.surface);
	const float queuePriorities[] = { 1 };
	const CreateQueues createQueues[] = { {vk.queueFamily, queuePriorities} };
	const char* deviceExtensionNames[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
	vk.device = createVulkanDevice(vk.physicalDevice, createQueues, deviceExtensionNames);

	create_swapChain_doubleBuff(vk.swapchainDb, vk.physicalDevice, vk.device, vk.surface);

	const VkAttachmentDescription rpAttachmentDesc = {
		.format = VK_FORMAT_B8G8R8A8_SRGB,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, // this informs the driver what layout to expect at the beginning of the renderPass
		.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // this tells the driver to perform a layout transition at the end of the renderPass
	};
	vk.renderPass = createSimpleRenderPass({&rpAttachmentDesc, 1}, false);

	for (int i = 0; i < 2; i++) {
		vk.framebuffers[i] = createFrameBuffer(vk.device, vk.renderPass,
			{&vk.swapchainDb.imageViews[i], 1},
			vk.swapchainDb.w, vk.swapchainDb.h);
	}

	const VkDescriptorSetLayoutBinding descriptorSetBindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1, // would be more than 1 for arrays of buffers
			.stageFlags = VK_SHADER_STAGE_ALL,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	vk.descriptorSetLayout = createDescriptorSetLayout(vk.device, descriptorSetBindings);
	vk.pipelineLayout = createPipelineLayout(vk.device, {&vk.descriptorSetLayout, 1});

	const auto vertShad = loadSpivShaderModule(vk.device, SHADERS_FOLDER"example_2_vert.glsl");
	const auto fragShad = loadSpivShaderModule(vk.device, SHADERS_FOLDER"example_2_frag.glsl");
	const VkPipelineShaderStageCreateInfo shaderStages[] = {
		makeShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, vertShad),
		makeShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, fragShad),
	};

	const VkVertexInputBindingDescription vertexInputBindings[] = { {
		.binding = 0,
		.stride = 5 * sizeof(float),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	} };

	const VkVertexInputAttributeDescription vertexInputAttribs[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 0,
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = 3*sizeof(float),
		},
	};

	vk.pipeline = createGraphicsPipeline(vk.device,
		shaderStages,
		vertexInputBindings, vertexInputAttribs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		1, false,
		vk.pipelineLayout, vk.renderPass, 0);

	vk.cmdPool = createCmdPool(vk.device, vk.queueFamily);


}