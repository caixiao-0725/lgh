#pragma once
#include"origin.h"
#include"cyPointCloud.h"

#define RADIUS 0.005f


struct Node {
	Node() : position(0, 0, 0), color(0, 0, 0), weight(0), firstChild(-1) {}
	Vec3f position;
	Color color;
	Vec3f stdev;   //标准差 
	float weight;
	int firstChild;
	void AddLight(float w,  Vec3f& p, Color& c)
	{
		weight += w;
		position += w * p;
		color += w * c;
		stdev += w * (p * p);
	}
	void Normalize()
	{
		if (weight > 0) {
			position /= weight;
			stdev = stdev / weight - position * position;
		}
	}
};

struct Level {
	Level() : colors(nullptr), pDev(nullptr) {}
	~Level() { delete[] colors; delete[] pDev; }
	cy::PointCloud<Vec3f, float, 3, int> pc;
	Color* colors;
	Vec3f* pDev; // position deviation for random shadow sampling
};

struct BVHNode {
	int left, right;    // 左右子树索引
	int n, index;       // 叶子节点信息               
	vec3f AA, BB;        // 碰撞盒
};

struct Material {
	vec3f emission;          // 作为光源时的发光颜色
	vec3f baseColor;
};

struct hitresult {
	bool IsHit;
	float distance;
	vec3f hitPoint;          // 光线命中点
	vec3f hitNormal;           // 命中点法线
	bool isInside;
	Material material;
};

struct Sphere {
	vec3f position;
	float radius;
	Material material;
};

struct Triangle {
	vec3f p1, p2, p3;    // 顶点坐标
	vec3f n;    // 顶点法线
	//Material material;  // 材质
};

struct BallLight {
	vec3f position;
	vec3f emission;
	vec3f velocity;
	vec3f baseColor;
};

struct Ray {
	vec3f origin;
	vec3f direction;
	//__device__ Ray():origin(vec3f(0, 0, 0)),direction(vec3f(1, 0, 0)) {}
	__device__ hitresult hitTriangle(Triangle triangle) {
		hitresult res;
		res.distance = 100000.0f;
		res.IsHit = false;
		res.isInside = false;

		vec3f p1 = triangle.p1;
		vec3f p2 = triangle.p2;
		vec3f p3 = triangle.p3;
			
		vec3f S = origin;    // 射线起点
		vec3f d = direction;     // 射线方向
		vec3f N = normalize((p2 - p1)^( p3 - p1));    // 法向量

		// 从三角形背后（模型内部）击中
		if (dot(N, d) > 0.0f) {
			N = -1.0f*N;
			res.isInside = true;
		}

		// 如果视线和三角形平行
		if (abs(dot(N, d)) < 0.00001f) return res;

		// 距离
		float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
		if (t < 0.0005f) return res;    // 如果三角形在光线背面

		// 交点计算
		vec3f P = S + d * t;

		// 判断交点是否在三角形中
		vec3f c1 = (p2 - p1)^( P - p1);
		vec3f c2 = (p3 - p2)^( P - p2);
		vec3f c3 = (p1 - p3)^( P - p3);
		bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
		bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

		// 命中，封装返回结果
		if (r1 || r2) {
			res.IsHit = true;
			res.hitPoint = P;
			res.distance = t;
		}

		return res;
	}
	__device__ hitresult hit(Sphere s) {
		hitresult res;
		res.IsHit = false;
		vec3f d1 = origin - s.position;
		vec3f dir = direction / direction.Length();
		res.material = s.material;
		//res.hitView = dir;
		float a = dir.dot(dir);
		float b = 2 * d1.dot(dir);
		float c = d1.dot(d1) - s.radius * s.radius;
		float delta = b * b - 4 * a * c;
		if (delta > 0) {
			float t1 = (-b + sqrt(delta)) / (2 * a);
			float t2 = (-b - sqrt(delta)) / (2 * a);
			if (t2 > 0) {
				res.IsHit = true;
				res.distance = t2;
				res.hitPoint = origin + direction * (t2 - 0.00001);
				res.hitNormal = origin + direction * (t2 - 0.00001) - s.position;
				res.hitNormal /= res.hitNormal.Length();
			}
			else if (t1 > 0) {
				res.IsHit = true;
				res.distance = t1;
				res.hitPoint = origin + direction * (t1 - 0.00001);
				res.hitNormal = origin + direction * (t1 - 0.00001) - s.position;
				res.hitNormal /= res.hitNormal.Length();
			}
		}
		return res;
	}

	__device__ hitresult hit(BallLight s) {
		hitresult res;
		res.IsHit = false;
		vec3f d1 = origin - s.position;
		vec3f dir = direction / direction.Length();
		res.material.emission = s.emission;
		res.material.baseColor = s.baseColor;
		//res.hitView = dir;
		float a = dir.dot(dir);
		float b = 2 * d1.dot(dir);
		float c = d1.dot(d1) - RADIUS * RADIUS;
		float delta = b * b - 4 * a * c;
		if (delta > 0) {
			float t1 = (-b + sqrt(delta)) / (2 * a);
			float t2 = (-b - sqrt(delta)) / (2 * a);
			if (t2 > 0) {
				res.IsHit = true;
				res.distance = t2;
				res.hitPoint = origin + direction * (t2 - 0.00001);

			}
			else if (t1 > 0) {
				res.IsHit = true;
				res.distance = t1;
				res.hitPoint = origin + direction * (t1 - 0.00001);
			}
		}
		return res;
	}

	__device__ float hitAABB(vec3f AA, vec3f BB) {
		vec3f invdir = 1.0f / direction;

		vec3f f = (BB - origin) * invdir;
		vec3f n = (AA - origin) * invdir;

		vec3f tmax = Max(f, n);
		vec3f tmin = Min(f, n);

		float t1 = fminf(tmax.x, fminf(tmax.y, tmax.z));
		float t0 = fmaxf(tmin.x, fmaxf(tmin.y, tmin.z));

		return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1.0f);
	}
	// 遍历 BVH 求交
	__device__ hitresult hitBVH(BVHNode* bvh,BallLight* balls) {
		hitresult res;
		res.IsHit = false;
		res.distance = 10000.0f;

		// 栈
		int stack[256];
		int sp = 0;

		stack[sp++] = 0;
		while (sp > 0) {
			int top = stack[--sp];
			BVHNode node = bvh[top];

			// 是叶子节点，遍历三角形，求最近交点
			if (node.n > 0) {
				int L = node.index;
				int R = node.index + node.n - 1;
				hitresult r;
				r.IsHit = false;
				r.distance = 10000.0f;
				for (int i = L; i < R + 1; i++) {
					hitresult r_temp = hit(balls[i]);
					if (r_temp.IsHit && r_temp.distance < r.distance) r = r_temp;
				}
				
				if (r.IsHit && r.distance < res.distance) res = r;
				continue;
			}

			// 和左右盒子 AABB 求交
			float d1 = 10000.0f; // 左盒子距离
			float d2 = 10000.0f; // 右盒子距离
			if (node.left > 0) {
				BVHNode leftNode = bvh[node.left];
				d1 = hitAABB(leftNode.AA, leftNode.BB);
			}
			if (node.right > 0) {
				BVHNode rightNode = bvh[node.right];
				d2 = hitAABB(rightNode.AA, rightNode.BB);
			}

			// 在最近的盒子中搜索
			if (d1 > 0 && d2 > 0) {
				if (d1 < d2) { // d1<d2, 左边先
					stack[sp++] = node.right;
					stack[sp++] = node.left;
				}
				else {    // d2<d1, 右边先
					stack[sp++] = node.left;
					stack[sp++] = node.right;
				}
			}
			else if (d1 > 0) {   // 仅命中左边
				stack[sp++] = node.left;
			}
			else if (d2 > 0) {   // 仅命中右边
				stack[sp++] = node.right;
			}
		}

		return res;
	}

	__device__ hitresult hitBVH(BVHNode* bvh,Triangle* triangles) {
		hitresult res;
		res.IsHit = false;
		res.distance = 10000.0f;

		// 栈
		int stack[256];
		int sp = 0;

		stack[sp++] = 0;
		while (sp > 0) {
			int top = stack[--sp];
			BVHNode node = bvh[top];

			// 是叶子节点，遍历三角形，求最近交点
			if (node.n > 0) {
				int L = node.index;
				int R = node.index + node.n - 1;
				hitresult r;
				r.IsHit = false;
				r.distance = 10000.0f;
				for (int i = L; i < R + 1; i++) {
					hitresult r_temp = hitTriangle(triangles[i]);
					if (r_temp.IsHit && r_temp.distance < r.distance) r = r_temp;
				}
				if (r.IsHit && r.distance < res.distance) res = r;
				continue;
			}

			// 和左右盒子 AABB 求交
			float d1 = 10000.0f; // 左盒子距离
			float d2 = 10000.0f; // 右盒子距离
			if (node.left > 0) {
				BVHNode leftNode = bvh[node.left];
				d1 = hitAABB(leftNode.AA, leftNode.BB);
			}
			if (node.right > 0) {
				BVHNode rightNode = bvh[node.right];
				d2 = hitAABB(rightNode.AA, rightNode.BB);
			}

			// 在最近的盒子中搜索
			if (d1 > 0 && d2 > 0) {
				if (d1 < d2) { // d1<d2, 左边先
					stack[sp++] = node.right;
					stack[sp++] = node.left;
				}
				else {    // d2<d1, 右边先
					stack[sp++] = node.left;
					stack[sp++] = node.right;
				}
			}
			else if (d1 > 0) {   // 仅命中左边
				stack[sp++] = node.left;
			}
			else if (d2 > 0) {   // 仅命中右边
				stack[sp++] = node.right;
			}
		}

		return res;
	}
	__device__ hitresult hitWalls(vec3f AA, vec3f BB) {
		hitresult res;
		res.IsHit = false;
		vec3f invdir = 1.0f / direction;

		vec3f f = (BB - origin) * invdir;
		vec3f n = (AA - origin) * invdir;

		vec3f tmax = Max(f, n);
		vec3f tmin = Min(f, n);

		float t1 = fminf(tmax.x, fminf(tmax.y, tmax.z));
		float t0 = fmaxf(tmin.x, fmaxf(tmin.y, tmin.z));
		if (t1 >= t0) {
			res.IsHit = true;
			res.distance = (t0 > 0.0) ? (t0) : (t1);
			res.hitPoint = origin + res.distance * direction;
		}
		else {
			res.distance = 10000.0f;
		}

		return res;
	}
};
__device__ void atomicVecAdd(vec3f& a,vec3f b) {
	atomicAdd(&(a.x), b.x); 
	atomicAdd(&(a.y), b.y); 
	atomicAdd(&(a.z), b.z);
}

__device__ void Collide_BalltoBall(vec3f& p, vec3f& v, vec3f& p_, vec3f& v_) {
	//位置更新
	vec3f en = normalize(p_ - p);
	float d = RADIUS - (p - p_).Length()/2.0f;
	atomicVecAdd(p, -d * en);
	atomicVecAdd(p_, d * en);
	//速度更新
	vec3f delte_v = v - v_;
	vec3f vn = delte_v.dot(en) * en;
	atomicVecAdd(v, - 1.5f * vn );
	atomicVecAdd(v_ , 1.5f * vn );
}


__device__ void Collide_Walls(vec3f& p, vec3f& v, vec3f AA, vec3f BB) {
	for (int i = 0; i < 3; i++) {
		if (p.raw[i] <= AA.raw[i]) {
			float d = AA.raw[i] - p.raw[i];
			if (v.raw[1] < 0.0001f && v.raw[1]>-0.05f) {
				v = 0.0f;
				if (d > 0)p.raw[i] = AA.raw[i] - 0.99f * RADIUS;
				else p.raw[i] = AA.raw[i] + 0.99f * RADIUS;
				
			}
			else {
				p.raw[i] += 1.01f * d;
				v.raw[i] *= -0.3f;
			}
			
		}
	}
	for (int i = 0; i < 3; i++) {
		if (p.raw[i] >= BB.raw[i]) {
			float d = BB.raw[i] - p.raw[i];
			if (v.raw[1] < 0.0001f && v.raw[1]>-0.05f) {
				v = 0.0f;
				if (d > 0)p.raw[i] = BB.raw[i] - 0.99f * RADIUS;
				else p.raw[i] = BB.raw[i] + 0.99f * RADIUS;

			}
			else {
				p.raw[i] += 1.01f * d;
				v.raw[i] *= -0.3f;
			}
			
		}
	}
}

__device__ bool BallhitAABB(vec3f p, vec3f AA, vec3f BB) {
	return (p.x > (AA.x - RADIUS) && p.x < (BB.x + RADIUS)) && (p.y > (AA.y - RADIUS) && p.y < (BB.y + RADIUS)) && (p.z > (AA.z - RADIUS) && p.z < (BB.z + RADIUS));
}

__device__ bool BallhitTriangles(vec3f& p, vec3f& v, Triangle triangles) {
	vec3f d1 = triangles.p1 - p;
	vec3f n = triangles.n;
	vec3f d = -1.0f * n;
	float t = d1.dot(d);
	if (t<-RADIUS || t>RADIUS) return false;
	vec3f P = p + d * t;

	// 判断交点是否在三角形中
	vec3f c1 = (triangles.p2 - triangles.p1) ^ (P - triangles.p1);
	vec3f c2 = (triangles.p3 - triangles.p2) ^ (P - triangles.p2);
	vec3f c3 = (triangles.p1 - triangles.p3) ^ (P - triangles.p3);
	bool r1 = (dot(c1, n) > 0 && dot(c2, n) > 0 && dot(c3, n) > 0);
	bool r2 = (dot(c1, n) < 0 && dot(c2, n) < 0 && dot(c3, n) < 0);
	// 命中面，更新位置
	if (r1 || r2) {
		float distance = RADIUS - t;
		if (v.raw[1] < 0.0001f && v.raw[1]>-0.05f) {
			v = 0.0f;
			p = P + 0.98f * RADIUS * n;
		}
		else {
			v -= 1.4f * (n.dot(v)) * n;
			p += 1.01f * distance * n;
		}
		return true;
	}
	//还没命中,看球和三条线段有没有交点
	return false;

}

__device__ void Collide_Model(vec3f& p, vec3f& v, BVHNode* bvh, Triangle* triangles) {
	// 栈
	int stack[256];
	int sp = 0;

	stack[sp++] = 0;
	while (sp > 0) {
		int top = stack[--sp];
		BVHNode node = bvh[top];

		// 是叶子节点，遍历三角形，求最近交点
		if (node.n > 0) {
			int L = node.index;
			int R = node.index + node.n - 1;
			bool r_temp = false;
			for (int i = L; i < R + 1; i++) {
				bool r_temp = BallhitTriangles(p, v,triangles[i]);
			}
		}

		// 和左右盒子 AABB 求交
		bool d1 = false; // 左盒子
		bool d2 = false; // 右盒子距离
		if (node.left > 0) {
			BVHNode leftNode = bvh[node.left];
			d1 = BallhitAABB(p,leftNode.AA, leftNode.BB);
		}
		if (node.right > 0) {
			BVHNode rightNode = bvh[node.right];
			d2 = BallhitAABB(p,rightNode.AA, rightNode.BB);
		}

		// 在最近的盒子中搜索
		if (d1 && d2) {
			stack[sp++] = node.right;
			stack[sp++] = node.left;
		}
		else if (d1) {   // 仅命中左边
			stack[sp++] = node.left;
		}
		else if (d2) {   // 仅命中右边
			stack[sp++] = node.right;
		}
	}
}