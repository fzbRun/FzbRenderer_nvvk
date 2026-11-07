#include "renderer/NeuralTextureCompression/NeuralTextureCompressionShaderio.h"

[Differentiable]
float square(float x, float y){
    return x * x + y * y;
}
//----------------------------------------------
[Differentiable]
float square_debug(float x, float y){
    let x2 = x * x;
    let y2 = debugGrad(y * y);
    return x2 + y2;
}

float debugGrad(float x) { return x; }
[BackwardDerivativeOf(debugGrad)]
void debugGradBwd(inout DifferentialPair<float> x, float dOut){
    printf("Gradient of y2 is %f\n", dOut);
    x = diffPair(x.p, dOut);
}
//----------------------------------------------
struct Point : IDifferentiable {
    float x;
    float y;
}

[Differentiable]
float square_struct(Point p){
    return p.x * p.x + p.y * p.y;
}
//----------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain_AutoDiff(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    //var xf = diffPair(3.0f, 1.0f);
    //var yf = diffPair(4.0f, 1.0f);
    //let fwd = fwd_diff(square_debug)(xf, yf);

    var p = diffPair(Point(3.0f, 4.0f), Point(1.0f, 1.0f));
    let fwd = fwd_diff(square_struct)(p);

    printf("square Result: %f\n", square_debug(3.0f, 4.0f));
    printf("diff Result: %f\n", fwd.d);

    let dldSquare = 1.0f;
    bwd_diff(square_struct)(p, dldSquare);
    printf("diff b Result: %f %f\n", p.d.x, p.d.y);

    //var xb = diffPair(3.0f, 0.0f);
    //var yb = diffPair(4.0f, 0.0f);
    //bwd_diff(square_debug)(xb, yb, dldSquare);
    //printf("diff b Result: %f %f\n", xb.d, yb.d);
}