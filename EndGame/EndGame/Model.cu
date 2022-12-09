#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "Model.cuh"
#include<algorithm>

//���캯�������������.obj�ļ�·��
Model::Model(const char* filename) : verts_(), faces_() {
    float maxx = 0;
    float minx = 0;
    float maxy = 0;
    float miny = 0;
    float maxz = 0;
    float minz = 0;
    std::ifstream in;
    in.open(filename, std::ifstream::in);//��.obj�ļ�
    if (in.fail()) return;
    std::string line;
    while (!in.eof()) {//û�е��ļ�ĩβ�Ļ�
        std::getline(in, line);//����һ��
        std::istringstream iss(line.c_str());
        char trash;
        if (!line.compare(0, 2, "v ")) {//�����һ�е�ǰ�����ַ��ǡ�v ���Ļ��������Ƕ�������
            iss >> trash;
            vec3f v;//���붥������
            for (int i = 0; i < 3; i++) iss >> v.raw[i];
            verts_.push_back(v);//���붥�㼯
            maxx = max(maxx, v.x); maxy = max(maxy, v.y); maxz = max(maxz, v.z);
            minx = min(minx, v.x); miny = min(miny, v.y); minz = min(minz, v.z);
        }
        else if (!line.compare(0, 2, "f ")) {//�����һ�е�ǰ�����ַ��ǡ�f ���Ļ�����������Ƭ����
            vec3i v;
            int iuv, idx;//idx�Ƕ���������itrash������������ʱ�ò�������������ͷ�������
            iss >> trash;
            int i = 0;
            while (iss >> idx >> trash >> iuv >> trash >> iuv) {//��ȡx/x/x��ʽ
                idx--; // all indices start at 1, not 0
                v.raw[i] = idx;;//�������Ƭ�Ķ��㼯
                i++;
            } 
            faces_.push_back(v);//�Ѹ���Ƭ����ģ�͵���Ƭ��          
        }    
    }
    // ģ�ʹ�С��һ��
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


    std::cout << "v# " << verts_.size() << "   f# " << faces_.size() << std::endl;//�����������Ƭ����

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