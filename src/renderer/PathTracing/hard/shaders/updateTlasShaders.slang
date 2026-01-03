#include "renderer/PathTracing/hard/shaderio.h"

[[vk::push_constant]] ConstantBuffer<PathTracingPushConstant, ScalarDataLayout> pushConst;