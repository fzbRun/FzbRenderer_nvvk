#include "renderer/PathTracing/hard/shaders/pathTracingCommon.slang"
#include "renderer/SVOPathGuiding/hard/shaderio.h"


[[vk::push_constant]] ConstantBuffer<SVOPathGuidingPushConstant, ScalarDataLayout> pushConst;