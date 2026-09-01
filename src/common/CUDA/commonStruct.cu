#include "./commonStruct.cuh"

FzbAABB::FzbAABB() {};
FzbAABB::FzbAABB(float leftX, float rightX, float leftY, float rightY, float leftZ, float rightZ) {
	this->leftX = leftX;
	this->rightX = rightX;
	this->leftY = leftY;
	this->rightY = rightY;
	this->leftZ = leftZ;
	this->rightZ = rightZ;
}
/*
uint32_t floatToUintBits(float value) {
	uint32_t result;
	std::memcpy(&result, &value, sizeof(result));
	return result;
}
FzbAABBUint::FzbAABBUint() {
	this->leftX = floatToUintBits(FLT_MAX);
	this->leftY = floatToUintBits(FLT_MAX);
	this->leftZ = floatToUintBits(FLT_MAX);
	this->rightX = floatToUintBits(-FLT_MAX);
	this->rightY = floatToUintBits(-FLT_MAX);
	this->rightZ = floatToUintBits(-FLT_MAX);
}
*/