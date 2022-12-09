#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "Model.cuh"
#include<algorithm>

//构造函数，输入参数是.obj文件路径
Model::Model(const char* filename) : verts_(), faces_() {
    float maxx = 0;
    float minx = 0;
    float maxy = 0;
    float miny = 0;
    float maxz = 0;
    float minz = 0;
    std::ifstream in;
    in.open(filename, std::ifstream::in);//打开.obj文件
    if (in.fail()) return;
    std::string line;
    while (!in.eof()) {//没有到文件末尾的话
        std::getline(in, line);//读入一行
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v ")) {//如果这一行的前两个字符是“v ”的话，代表是顶点数据
            iss >> trash;
            vec3f v;//读入顶点坐标
            for (int i = 0; i < 3; i++) iss >> v.raw[i];
            verts_.push_back(v);//加入顶点集
            maxx = max(maxx, v.x); maxy = max(maxy, v.y); maxz = max(maxz, v.z);
            minx = min(minx, v.x); miny = min(miny, v.y); minz = min(minz, v.z);
        }
        else if (!line.compare(0, 2, "f ")) {//如果这一行的前两个字符是“f ”的话，代表是面片数据
            vec3i v;
            int iuv, idx;//idx是顶点索引，itrash用来读我们暂时用不到的纹理坐标和法线向量
            iss >> trash;
            int i = 0;
            while (iss >> idx >> trash >> iuv >> trash >> iuv) {//读取x/x/x格式
                idx--; // all indices start at 1, not 0
                v.raw[i] = idx;;//加入该面片的顶点集
                i++;
            } 
            faces_.push_back(v);//把该面片加入模型的面片集          
        }    
    }
    // 模型大小归一化
    float lenx = maxx - minx;
    float leny = maxy - miny;
    float lenz = maxz - minz;
    float maxaxis = max(lenx, max(leny, lenz))/2;
    for (auto& v : verts_) {
        float temp = v.z;
        v.x = 0.8f*(v.x - minx) / maxaxis -1.0f ;
        v.z = 0.8f * (v.y - miny) / maxaxis -1.0f;
        v.y = -1.0f*(temp - minz) / maxaxis+lenz/maxaxis +0.1f ;
    }


    std::cout << "v# " << verts_.size() << "   f# " << faces_.size() << std::endl;//输出顶点与面片数量

}



Model::~Model() {
}

int Model::nverts() {
    return (int)verts_.size();
}

int Model::nfaces() {
    return (int)faces_.size();
}

vec3i Model::face(int i) {
    return faces_[i];
}


vec3f Model::vert(int i) {
    return verts_[i];
}