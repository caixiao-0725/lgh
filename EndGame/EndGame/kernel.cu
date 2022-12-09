
#include<string>
#include<fstream>
#include <iostream>
#include<sstream>
#include<time.h>

#include<cuda.h>
#include <cuda_runtime.h>
#include"device_launch_parameters.h"


#include"origin.h"
#include"Collide.h"
#include<vector>
#include <algorithm>  
#include "Model.cuh"
#include<time.h>
#include "svpng.inc"

#define PI 3.1415926f
#define BallNum (1024*16)
#define RADIUS 0.005f
#define t 0.05f
#define INF 114514.0f
#define BASECOLOR 0.2f

typedef unsigned int uint;



// settings
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 512;




clock_t start, end_time;
float gap_time;





__device__ vec3f lightingFunction(int level, int light_id, Vec3f& light_position,Color& light_intensity,vec3f pos) {
    //if (level == 0 && (light_position - pos).Length() < 1.1f*RADIUS) return vec3f();
    vec3f res = light_intensity / (light_position - pos).LengthSquared()*0.0005f;
    return res;
}

__device__ vec3f Light(Vec3f& position,float  alpha,Level* levels,int numLevels,float cellSize)
{
    vec3f res;
    if (numLevels > 1) {

        // First level
        float r = alpha * cellSize;
        float rr = r * r;
        
        levels[0].pc.GetPoints(position, r * 2, [&](int i, Vec3f& p, float dist2, float& radius2) {
            Color c = levels[0].colors[i];
            if (dist2 > rr) c *= 1 - (sqrtf(dist2) - r) / r;
            res += lightingFunction(0, i, p, c, position);
            });
        

        // Middle levels
        for (int level = 1; level < numLevels - 1; level++) {
            float r_min = r;
            float rr_min = r * r;
            r *= 2;
            levels[level].pc.GetPoints(position, r * 2, [&](int i, Vec3f& p, float dist2, float& radius2) {
                if (dist2 <= rr_min) return ;
                float d = sqrtf(dist2);
                if (i > 100000) return;
                Color c = levels[level].colors[i];
                
                if (d > r) c *= 1 - (d - r) / r;
                else c *= (d - r_min) / r_min;
                res += lightingFunction(level, i, p, c, position);
                });
            
        }
        
        // Last level
        float r_min = r;
        float rr_min = r * r;
        r *= 2;
        rr = r * r;
        int n = levels[numLevels - 1].pc.GetPointCount();
        for (int i = 0; i < n; i++) {
            Vec3f& p = levels[numLevels - 1].pc.GetPoint(i);
            float dist2 = (position - p).LengthSquared();
            if (dist2 <= rr_min) continue;
            int id = levels[numLevels - 1].pc.GetPointIndex(i);
            Color c = levels[numLevels - 1].colors[id];
            if (dist2 < rr) c *= (sqrtf(dist2) - r_min) / r_min;
            res += lightingFunction(numLevels, i, p, c, position);
        }

    }
    else {
        // Single-level (a.k.a. brute-force)
        int n = levels[0].pc.GetPointCount();
        for (int i = 0; i < n; i++) {
            Vec3f& p = levels[0].pc.GetPoint(i);
            int id = levels[0].pc.GetPointIndex(i);
            Color c = levels[0].colors[id];
            res += lightingFunction(0, i, p, c, position);
        }
    }
    return res;
}

__global__ void Update_(BallLight* balls, BVHNode* bvh,Triangle* triangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > BallNum) return;
    balls[i].position += balls[i].velocity * t;
    balls[i].velocity += vec3f(0.0f, -1.0f, 0.0f) * t;
    balls[i].velocity *= 0.98f;
    int j = i + 1;

    while ((balls[i].position.x - balls[j].position.x) <= 2 * RADIUS) {
        if ((balls[i].position - balls[j].position).Length() <= 2 * RADIUS) {
            Collide_BalltoBall(balls[i].position, balls[i].velocity, balls[j].position, balls[j].velocity);
        }
        j++;
        if (j > BallNum - 1)break;
    }
    Collide_Walls(balls[i].position, balls[i].velocity, vec3f(-1.1f, 0.0f, -1.5f), vec3f(1.1f, 5.0f, 3.0f));
    Collide_Model(balls[i].position, balls[i].velocity, bvh, triangles);


}

__global__ void testKernal(unsigned char* output,int width, int height,BallLight* balls,BVHNode* bvh, Level* levels, int numLevels, float cellSize, BVHNode* model_bvh,Triangle* triangles)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        vec3f answer;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                vec3f ro = vec3f(-0.5f, 1.0f, 0.9f);
                vec3f rd_ = vec3f(0.28, -0.4f, -1) + ((float)(x+i * 0.5f) / SCR_WIDTH - 0.5f) * vec3f(1.5, 0, 0) + ((float)(y+i*0.5f) - SCR_HEIGHT / 2) / SCR_WIDTH * vec3f(0, 1.5, 0);
                vec3f rd = normalize(rd_);
                Ray ray = { ro, rd };
                hitresult res;
                res.IsHit = false;
                res.distance = 10000.0f;
                hitresult temp_res; 
                temp_res.IsHit = false;
                temp_res.distance = 10000.0f;




                temp_res = ray.hitWalls(vec3f(-1.1f,0.0f,-2.1f),vec3f(1.1f,5.0f,3.0f));
                if (temp_res.IsHit) {
                    res = temp_res;
                    res.material.baseColor = vec3f(BASECOLOR);
                }
                temp_res = ray.hitBVH(bvh,balls);
                if (temp_res.IsHit && temp_res.distance < res.distance) {
                    res = temp_res;
                }
                temp_res = ray.hitBVH(model_bvh, triangles);
                if (temp_res.IsHit && temp_res.distance < res.distance) {
                    res = temp_res;
                    res.material.baseColor = vec3f(BASECOLOR);
                }
        



                //我们已经找到了碰撞点，那么接下来就对碰撞点进行光照的计算
                if (res.IsHit) {
                    vec3f pos = res.hitPoint;
                    vec3f col = res.material.baseColor *Light(pos, 1.0f, levels, numLevels, cellSize);
                    answer += col;


                }
                else {
                    vec3f col = vec3f(0.0f);
                    answer += col;
                }
            }
        }
        answer /= 4.0f;
        answer.clamp(0.0f, 1.0f);
        int index = (SCR_HEIGHT - y - 1) * SCR_WIDTH + x;
        output[4 * index] = unsigned char(answer.x*255);
        output[4 * index + 1] = unsigned char(answer.y * 255);
        output[4 * index + 2] = unsigned char(answer.z * 255);
        output[4 * index + 3] = 255;

    }
}

// 按照三角形中心排序 -- 比较函数
bool cmpx(const BallLight& t1, const BallLight& t2) {
    return t1.position.x < t2.position.x;
}
bool cmpy(const BallLight& t1, const BallLight& t2) {
    return t1.position.y < t2.position.y;
}
bool cmpz(const BallLight& t1, const BallLight& t2) {
    return t1.position.z < t2.position.z;
}

bool cmpx_(Triangle& t1,  Triangle& t2) {
    vec3f center1 = (t1.p1 + t1.p2 + t1.p3) / vec3f(3, 3, 3);
    vec3f center2 = (t2.p1 + t2.p2 + t2.p3) / vec3f(3, 3, 3);
    return center1.x < center2.x;
}
bool cmpy_(Triangle& t1, Triangle& t2) {
    vec3f center1 = (t1.p1 + t1.p2 + t1.p3) / vec3f(3, 3, 3);
    vec3f center2 = (t2.p1 + t2.p2 + t2.p3) / vec3f(3, 3, 3);
    return center1.y < center2.y;
}
bool cmpz_(Triangle& t1, Triangle& t2) {
    vec3f center1 = (t1.p1 + t1.p2 + t1.p3) / vec3f(3, 3, 3);
    vec3f center2 = (t2.p1 + t2.p2 + t2.p3) / vec3f(3, 3, 3);
    return center1.z < center2.z;
}

int buildBVH(std::vector<BallLight>& balls_host, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;
    nodes.push_back(BVHNode());
    int id = nodes.size() - 1;
    nodes[id].AA = vec3f(3, 3, 3);
    nodes[id].BB = vec3f(-3, -3, -3);
    nodes[id].n = 0;
    nodes[id].index = l;
    nodes[id].left = -1;
    nodes[id].right = -1;
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        nodes[id].AA.x = std::min(balls_host[i].position.x, nodes[id].AA.x);
        nodes[id].AA.y = std::min(balls_host[i].position.y, nodes[id].AA.y);
        nodes[id].AA.z = std::min(balls_host[i].position.z, nodes[id].AA.z);
        // 最大点 BB

        nodes[id].BB.x = std::max(balls_host[i].position.x, nodes[id].BB.x);
        nodes[id].BB.y = std::max(balls_host[i].position.y, nodes[id].BB.y);
        nodes[id].BB.z = std::max(balls_host[i].position.z, nodes[id].BB.z);
    }
    nodes[id].AA -= RADIUS;
    nodes[id].BB += RADIUS;
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }
    float lenx =nodes[id].BB.x - nodes[id].AA.x;
    float leny =nodes[id].BB.y - nodes[id].AA.y;
    float lenz =nodes[id].BB.z - nodes[id].AA.z;
    // 按 x 划分
    if (lenx >= leny && lenx >= lenz){
        std::sort(balls_host.begin() + l, balls_host.begin() + r + 1, cmpx);
    }
    // 按 y 划分
    else if (leny >= lenx && leny >= lenz)
        std::sort(balls_host.begin() + l, balls_host.begin() + r + 1, cmpy);
    // 按 z 划分
    else if (lenz >= lenx && lenz >= leny)
        std::sort(balls_host.begin() + l, balls_host.begin() + r + 1, cmpz);

    int mid = (l + r) / 2;
    nodes[id].right = buildBVH(balls_host, nodes ,mid + 1, r, n);
    nodes[id].left =  buildBVH(balls_host, nodes ,l, mid, n);
    
    return id;
}

// SAH 优化构建 BVH
int buildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    nodes.push_back(BVHNode());
    int id = nodes.size() - 1;
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3f(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3f(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].AA.x = min(nodes[id].AA.x, minx);
        nodes[id].AA.y = min(nodes[id].AA.y, miny);
        nodes[id].AA.z = min(nodes[id].AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].BB.x = max(nodes[id].BB.x, maxx);
        nodes[id].BB.y = max(nodes[id].BB.y, maxy);
        nodes[id].BB.z = max(nodes[id].BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }

    // 否则递归建树
    float Cost = INF;
    int Axis = 0;
    int Split = (l + r) / 2;
    for (int axis = 0; axis < 3; axis++) {
        // 分别按 x，y，z 轴排序
        if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx_);
        if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy_);
        if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz_);

        // leftMax[i]: [l, i] 中最大的 xyz 值
        // leftMin[i]: [l, i] 中最小的 xyz 值
        std::vector<vec3f> leftMax(r - l + 1, vec3f(-INF, -INF, -INF));
        std::vector<vec3f> leftMin(r - l + 1, vec3f(INF, INF, INF));
        // 计算前缀 注意 i-l 以对齐到下标 0
        for (int i = l; i <= r; i++) {
            Triangle& tri = triangles[i];
            int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

            leftMax[i - l].x = max(leftMax[i - l - bias].x, max(tri.p1.x, max(tri.p2.x, tri.p3.x)));
            leftMax[i - l].y = max(leftMax[i - l - bias].y, max(tri.p1.y, max(tri.p2.y, tri.p3.y)));
            leftMax[i - l].z = max(leftMax[i - l - bias].z, max(tri.p1.z, max(tri.p2.z, tri.p3.z)));

            leftMin[i - l].x = min(leftMin[i - l - bias].x, min(tri.p1.x, min(tri.p2.x, tri.p3.x)));
            leftMin[i - l].y = min(leftMin[i - l - bias].y, min(tri.p1.y, min(tri.p2.y, tri.p3.y)));
            leftMin[i - l].z = min(leftMin[i - l - bias].z, min(tri.p1.z, min(tri.p2.z, tri.p3.z)));
        }

        // rightMax[i]: [i, r] 中最大的 xyz 值
        // rightMin[i]: [i, r] 中最小的 xyz 值
        std::vector<vec3f> rightMax(r - l + 1, vec3f(-INF, -INF, -INF));
        std::vector<vec3f> rightMin(r - l + 1, vec3f(INF, INF, INF));
        // 计算后缀 注意 i-l 以对齐到下标 0
        for (int i = r; i >= l; i--) {
            Triangle& tri = triangles[i];
            int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

            rightMax[i - l].x = max(rightMax[i - l + bias].x, max(tri.p1.x, max(tri.p2.x, tri.p3.x)));
            rightMax[i - l].y = max(rightMax[i - l + bias].y, max(tri.p1.y, max(tri.p2.y, tri.p3.y)));
            rightMax[i - l].z = max(rightMax[i - l + bias].z, max(tri.p1.z, max(tri.p2.z, tri.p3.z)));
                                                                
            rightMin[i - l].x = min(rightMin[i - l + bias].x, min(tri.p1.x, min(tri.p2.x, tri.p3.x)));
            rightMin[i - l].y = min(rightMin[i - l + bias].y, min(tri.p1.y, min(tri.p2.y, tri.p3.y)));
            rightMin[i - l].z = min(rightMin[i - l + bias].z, min(tri.p1.z, min(tri.p2.z, tri.p3.z)));
        }

        // 遍历寻找分割
        float cost = INF;
        int split = l;
        for (int i = l; i <= r - 1; i++) {
            float lenx, leny, lenz;
            // 左侧 [l, i]
            vec3f leftAA = leftMin[i - l];
            vec3f leftBB = leftMax[i - l];
            lenx = leftBB.x - leftAA.x;
            leny = leftBB.y - leftAA.y;
            lenz = leftBB.z - leftAA.z;
            float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float leftCost = leftS * (i - l + 1);

            // 右侧 [i+1, r]
            vec3f rightAA = rightMin[i + 1 - l];
            vec3f rightBB = rightMax[i + 1 - l];
            lenx = rightBB.x - rightAA.x;
            leny = rightBB.y - rightAA.y;
            lenz = rightBB.z - rightAA.z;
            float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float rightCost = rightS * (r - i);

            // 记录每个分割的最小答案
            float totalCost = leftCost + rightCost;
            if (totalCost < cost) {
                cost = totalCost;
                split = i;
            }
        }
        // 记录每个轴的最佳答案
        if (cost < Cost) {
            Cost = cost;
            Axis = axis;
            Split = split;
        }
    }

    // 按最佳轴分割
    if (Axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx_);
    if (Axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy_);
    if (Axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz_);

    // 递归
    int left = buildBVHwithSAH(triangles, nodes, l, Split, n);
    int right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}



__global__ void SetPtr(Level* level,Color* color, Vec3f* pdev, cy::PointCloud<Vec3f, float, 3, int>::PointData* pd,int num) {
    int i = threadIdx.x;
    if (i < 1) {
        level[num].colors = color;
        level[num].pDev = pdev;
        level[num].pc.points = pd;
    }
}

int main()
{
    vec3f rainbow[7] = { vec3f(1.0f,0,0),vec3f(1.0f,0.647f,0.0f),vec3f(1.0f,1.0f,0),vec3f(0,1.0f,0),vec3f(0,0.498,1.0),vec3f(0,0,1.0f),vec3f(0.545f,0,1.0f) };
    std::vector<BallLight> balls_host(BallNum);

    int diff_num = 512;
    for (int i = 0; i < diff_num; i++) {
        balls_host[i].position = vec3f(2.0f * rand() / RAND_MAX - 1.0f, 0.5f * float(rand()) / RAND_MAX + 1.1f, -((0.5f * rand() / RAND_MAX) + 0.8f));
        balls_host[i].emission = 0.2f * rainbow[int(balls_host[i].position.x * 3.5 + 3.5f) % 7];
        balls_host[i].baseColor = vec3f(BASECOLOR * 0.5f);
        balls_host[i].velocity = vec3f(0.02f * rand() / RAND_MAX - 0.01f, -0.2f, 0);
    }

    for (int i = 0; i < BallNum - diff_num; i++) {
        balls_host[i + diff_num].position = vec3f(1.9f * rand() / RAND_MAX - 1.0f, 1.2f * float(rand()) / RAND_MAX + 1.6f, -((0.5f * rand() / RAND_MAX) + 0.8f));
        balls_host[i + diff_num].emission = 0.2f * rainbow[int(balls_host[i + diff_num].position.x * 3.5 + 3.5f) % 7];
        balls_host[i + diff_num].baseColor = vec3f(BASECOLOR * 0.5f);
        balls_host[i + diff_num].velocity = vec3f(0.02f * rand() / RAND_MAX - 0.01f, -0.2f, 0);
    }

    Model teapot = Model("Model/give.obj");
    std::vector<Triangle> triangles(teapot.nfaces());
    for (int i = 0; i < teapot.nfaces(); i++) {
        triangles[i].p1 = teapot.verts_[teapot.faces_[i].raw[0]];
        triangles[i].p2 = teapot.verts_[teapot.faces_[i].raw[1]];
        triangles[i].p3 = teapot.verts_[teapot.faces_[i].raw[2]];
        triangles[i].n = normalize((triangles[i].p2 - triangles[i].p1) ^ (triangles[i].p3 - triangles[i].p1));
    }

    std::vector<BVHNode> modelnodes;
    buildBVHwithSAH(triangles, modelnodes, 0, teapot.nfaces() - 1, 8);

    BVHNode* model_bvh_device;
    cudaMalloc((void**)&model_bvh_device, sizeof(BVHNode) * modelnodes.size());
    cudaMemcpy(model_bvh_device, modelnodes.data(), sizeof(BVHNode) * modelnodes.size(), cudaMemcpyHostToDevice);

    Triangle* tri_device;
    cudaMalloc((void**)&tri_device, sizeof(Triangle) * triangles.size());
    cudaMemcpy(tri_device, triangles.data(), sizeof(Triangle) * triangles.size(), cudaMemcpyHostToDevice);

    unsigned char output_host[SCR_HEIGHT*SCR_WIDTH* 4];
    int count = 0;
    // -----------
    while (count<500)
    {
        start = clock();;
        //获取碰撞对,先按照x排序
        std::sort(balls_host.begin(), balls_host.end()-1, cmpx);
        
        //物理位置更新
        BallLight* balls_device_update;
        
        cudaMalloc((void**)&balls_device_update, sizeof(BallLight) * BallNum);
        cudaMemcpy(balls_device_update, balls_host.data(), sizeof(BallLight) * BallNum, cudaMemcpyHostToDevice);
        Update_ << <BallNum / 1024, 1024 >> > (balls_device_update, model_bvh_device, tri_device);

        cudaMemcpy(balls_host.data(),balls_device_update, sizeof(BallLight) * BallNum, cudaMemcpyDeviceToHost);
   
        
        std::vector<BVHNode> bvhnodes ;

        buildBVH(balls_host, bvhnodes, 0, BallNum - 1, 8);

        // Compute the bounding box for the lighPoss
        vec3f boundMin = vec3f(bvhnodes[0].AA);
        vec3f boundMax = vec3f(bvhnodes[0].BB);
        vec3f boundDif = boundMax - boundMin;
        float boundDifMin = boundDif.Min();

        // Determine the actual highest level
        float highestCellSize;
        vec3i highestGridRes;
        float autoFitScale = 1.001f;
        highestCellSize = boundDif.Max() * autoFitScale;
        int s = int(1.0f / autoFitScale) + 2;
        if (s < 2) s = 2;
        highestGridRes = vec3i(s, s, s);

        int highestLevel = 8;
        int numLevels = highestLevel + 1;
        std::vector< std::vector<Node> > nodes(numLevels);
    
        auto gridIndex = [](IVec3i& index, Vec3f& pos, float cellSize)
        {
            Vec3f normP = pos / cellSize;
            index = IVec3i(int(normP.x), int(normP.y), int(normP.z));
            return normP - Vec3f(float(index.x), float(index.y), float(index.z));
        };

        auto addLightToNodes = [](std::vector<Node>& nds, int nodeIDs[8], Vec3f& interp, Vec3f& light_pos,Color& light_color)
        {
            for (int j = 0; j < 8; j++) {
                float w = ((j & 1) ? interp.x : (1 - interp.x)) * ((j & 2) ? interp.y : (1 - interp.y)) * ((j & 4) ? interp.z : (1 - interp.z));
                nds[nodeIDs[j]].AddLight(w, light_pos, light_color);
            }
        };


        // Generate the grid for the highest level
        Vec3f highestGridSize = Vec3f(highestGridRes.x - 1) * highestCellSize;
        Vec3f center = (boundMax + boundMin) / 2;
        Vec3f corner = center - highestGridSize / 2;
        nodes[highestLevel].resize(highestGridRes.x * highestGridRes.y * highestGridRes.z);

        for (int i = 0; i < BallNum; i++) {
            IVec3i index;
            Vec3f interp = gridIndex(index, balls_host[i].position - corner, highestCellSize);
            int is = index.z * highestGridRes.y * highestGridRes.x + index.y * highestGridRes.x + index.x;
            int nodeIDs[8] = {
                is,
                is + 1,
                is + highestGridRes.x,
                is + highestGridRes.x + 1,
                is + highestGridRes.x * highestGridRes.y,
                is + highestGridRes.x * highestGridRes.y + 1,
                is + highestGridRes.x * highestGridRes.y + highestGridRes.x,
                is + highestGridRes.x * highestGridRes.y + highestGridRes.x + 1,
            };
            for (int j = 0; j < 8; j++) assert(nodeIDs[j] >= 0 && nodeIDs[j] < (int)nodes[highestLevel].size());
            addLightToNodes(nodes[highestLevel], nodeIDs, interp, balls_host[i].position, balls_host[i].emission);
        }
        for (int i = 0; i < (int)nodes[highestLevel].size(); i++) nodes[highestLevel][i].Normalize();
        //最高层建立完毕  建立后面的层

        // Generate the lower levels
        float nodeCellSize = highestCellSize;
        IVec3i gridRes = highestGridRes;
        int levelSkip = 0;
        for (int level = highestLevel - 1; level > 0; level--) {
            // Find the number of nodes for this level
            int nodeCount = 0;
            for (int i = 0; i < (int)nodes[level + 1].size(); i++) {
                if (nodes[level + 1][i].weight > 0) {
                    nodes[level + 1][i].firstChild = nodeCount;
                    nodeCount += 8;
                }
            }

            if (nodeCount > BallNum / 4) {
                levelSkip = level;
                break;
            }

            nodes[level].resize(nodeCount);
            // Add the lights to the nodes
            nodeCellSize /= 2;
            gridRes *= 2;
            for (int i = 0; i < BallNum; i++) {
                IVec3i index;
                Vec3f interp = gridIndex(index, Vec3f(balls_host[i].position) - corner, nodeCellSize);
                // find the node IDs
                int nodeIDs[8];
                index = index<<(level + 2);
                for (int z = 0, j = 0; z < 2; z++) {
                    int iz = index.z + z;
                    for (int y = 0; y < 2; y++) {
                        int iy = index.y + y;
                        for (int x = 0; x < 2; x++, j++) {
                            int ix = index.x + x;
                            int hix = ix >> (highestLevel + 2);
                            int hiy = iy >> (highestLevel + 2);
                            int hiz = iz >> (highestLevel + 2);
                            int nid = hiz * highestGridRes.y * highestGridRes.x + hiy * highestGridRes.x + hix;
                            for (int l = highestLevel - 1; l >= level; l--) {
                                int ii = ((index.z >> l) & 4) | ((index.y >> (l + 1)) & 2) | ((index.x >> (l + 2)) & 1);
                                assert(nodes[l + 1][nid].firstChild >= 0);
                                nid = nodes[l + 1][nid].firstChild + ii;
                                assert(nid >= 0 && nid < (int)nodes[l].size());
                            }
                            nodeIDs[j] = nid;
                        }
                    }
                }
                addLightToNodes(nodes[level], nodeIDs, interp, balls_host[i].position, balls_host[i].emission);
            }
            for (int i = 0; i < (int)nodes[level].size(); i++) nodes[level][i].Normalize();
        }

        // Copy light data
        numLevels = highestLevel + 1 - levelSkip;
        int levelBaseSkip = 0;
        // Skip levels that have two few lights (based on minLevelLights).
        int minLevelLights = 8;

        for (int level = 1; level < numLevels; level++) {
            std::vector<Node>& levelNodes = nodes[level + levelSkip];
            int count = 0;
            for (int i = 0; i < (int)levelNodes.size(); i++) {
                if (levelNodes[i].weight > 0) {
                    count++;
                }
            }
            if (count < minLevelLights) {
                numLevels = level;
                break;
            }
        }

        //将levels传到gpu咋呀

        Level* levels_device; 

        Color** colors_device = new Color*[numLevels];
        Vec3f** pDevs_device = new Vec3f*[numLevels];
        cy::PointCloud<Vec3f, float, 3, int>::PointData** pointDatas_device = new cy::PointCloud<Vec3f, float, 3, int>::PointData * [numLevels];
    
        cudaMalloc((void**)&levels_device, sizeof(Level)* numLevels);

        Level* levels = new Level[numLevels];
        for (int level = 1; level < numLevels; level++) {
            std::vector<Node>& levelNodes = nodes[level + levelSkip];
            Level& thisLevel = levels[level];
            std::vector<Vec3f> pos(levelNodes.size());
            int lightCount = 0;
            for (int i = 0; i < (int)levelNodes.size(); i++) {
                if (levelNodes[i].weight > 0) {
                    pos[lightCount++] = levelNodes[i].position;
                }
            }

            thisLevel.pc.Build(lightCount, pos.data());
            thisLevel.colors = new Color[lightCount];
            thisLevel.pDev = new Vec3f[lightCount];
            for (int i = 0, j = 0; i < (int)levelNodes.size(); i++) {
                if (levelNodes[i].weight > 0) {
                    assert(j < lightCount);
                    thisLevel.colors[j] = levelNodes[i].color;
                    thisLevel.pDev[j].x = sqrtf(levelNodes[i].stdev.x) * PI;
                    thisLevel.pDev[j].y = sqrtf(levelNodes[i].stdev.y) * PI;
                    thisLevel.pDev[j].z = sqrtf(levelNodes[i].stdev.z) * PI;
                    j++;
                }
            }
            cudaMalloc((void**)&pDevs_device[level], sizeof(Vec3f) * lightCount);
            cudaMalloc((void**)&colors_device[level], sizeof(Color)* lightCount);
            cudaMalloc((void**)&pointDatas_device[level], sizeof(cy::PointCloud<Vec3f, float, 3, int>::PointData)* lightCount);

            cudaMemcpy(colors_device[level], thisLevel.colors, sizeof(Color)* lightCount, cudaMemcpyHostToDevice);
            cudaMemcpy(pDevs_device[level], thisLevel.pDev, sizeof(Vec3f)* lightCount, cudaMemcpyHostToDevice);
        
            cudaMemcpy(pointDatas_device[level], levels[level].pc.points, sizeof(cy::PointCloud<Vec3f, float, 3, int>::PointData)* lightCount, cudaMemcpyHostToDevice);
        
            levelNodes.resize(0);
            levelNodes.shrink_to_fit();
        }
        std::vector<Vec3f> pos(BallNum);
        levels[0].colors = new Color[BallNum];
        for (int i = 0; i < BallNum; i++) {
            pos[i] = balls_host[i].position;
            levels[0].colors[i] = balls_host[i].emission;
        }
        levels[0].pc.Build(BallNum, pos.data());
        cudaMemcpy(levels_device, levels, sizeof(Level)* numLevels, cudaMemcpyHostToDevice);

        cudaMalloc((void**)&pDevs_device[0], sizeof(Vec3f)* BallNum);
        cudaMalloc((void**)&colors_device[0], sizeof(Color)* BallNum);
        cudaMalloc((void**)&pointDatas_device[0], sizeof(cy::PointCloud<Vec3f, float, 3, int>::PointData)* BallNum);

        cudaMemcpy(colors_device[0], levels[0].colors, sizeof(Color)* BallNum, cudaMemcpyHostToDevice);
        cudaMemcpy(pDevs_device[0], levels[0].pDev, sizeof(Vec3f)* BallNum, cudaMemcpyHostToDevice);
        cudaMemcpy(pointDatas_device[0], levels[0].pc.points, sizeof(cy::PointCloud<Vec3f, float, 3, int>::PointData)* BallNum, cudaMemcpyHostToDevice);

        //布置gpu上的指针
        for (int i = 0; i < numLevels; i++) {
            SetPtr << <1, 1 >> > (levels_device, colors_device[i], pDevs_device[i], pointDatas_device[i], i);
        }

        BallLight* balls_device;
        cudaMalloc((void**)&balls_device, sizeof(BallLight)*BallNum);
        cudaMemcpy(balls_device, balls_host.data(), sizeof(BallLight) * BallNum, cudaMemcpyHostToDevice);

        BVHNode* nodes_device;
        cudaMalloc((void**)&nodes_device, sizeof(BVHNode) * bvhnodes.size());
        cudaMemcpy(nodes_device, bvhnodes.data(), sizeof(BVHNode) * bvhnodes.size(), cudaMemcpyHostToDevice);
        
        end_time = clock();;
        gap_time = (float)(end_time - start) / CLOCKS_PER_SEC;
        printf("准备时间：%f             ", gap_time);

        unsigned char* output_device;
        cudaMalloc((void**)&output_device, sizeof(unsigned char) * SCR_HEIGHT * SCR_WIDTH * 4);

        dim3 block(32, 32);
        dim3 grid((SCR_WIDTH-1) / block.x+1, (SCR_HEIGHT-1) / block.y+1);

        testKernal<<<grid,block>>>(output_device,SCR_WIDTH,SCR_HEIGHT,balls_device,nodes_device, levels_device, numLevels, nodeCellSize, model_bvh_device,tri_device);

        cudaMemcpy(output_host, output_device, sizeof(unsigned char)* SCR_HEIGHT* SCR_WIDTH * 4, cudaMemcpyDeviceToHost);

        //清理内存,随便清清啦

        for (int i = 0; i < numLevels; i++) {
            cudaFree(colors_device[i]);
            cudaFree(pDevs_device[i]);
            cudaFree(pointDatas_device[i]);
        }        
        cudaFree(balls_device);
        cudaFree(colors_device);
        cudaFree(nodes_device);
        cudaFree(levels_device);
        cudaFree(pDevs_device);
        cudaFree(pointDatas_device);
        delete [] levels;
        delete [] colors_device;
        delete[] pDevs_device;
        delete[] pointDatas_device;
        bvhnodes.clear();
        bvhnodes.shrink_to_fit();

        char filename[20];
        sprintf(filename, "png/%06d.png", count++);
        FILE* fp = fopen(filename, "wb");
        svpng(fp, SCR_WIDTH, SCR_HEIGHT, output_host, 1);
        fclose(fp);

        end_time = clock();
        gap_time = (float)(end_time - start) / CLOCKS_PER_SEC;
        printf("渲染时间:% f      当前帧数:%d\n ", gap_time,count);
        cudaFree(balls_device_update);
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    return 0;
}
