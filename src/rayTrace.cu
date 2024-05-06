#include<fstream>
#define SPHERES 20
#define rnd(x) ((x) * rand() / RAND_MAX)
#define DIM 512
#define INF 2e10f

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    __device__ float hit(float ox, float oy, float *n) const {
        float dx = ox - center.x;
        float dy = oy - center.y;
        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / radius;
            return dz + center.z;
        }
        return -INF;
    }
};



__constant__ Sphere spheres[SPHERES];

__global__ void kernel(unsigned char* ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        Sphere sphere = spheres[i];
        float n;
        float t = sphere.hit(ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = sphere.color.x * fscale;
            g = sphere.color.y * fscale;
            b = sphere.color.z * fscale;
            maxz = t;
        }
    }

    ptr[offset * 3 + 0] = (int)(r * 255);
    ptr[offset * 3 + 1] = (int)(g * 255);
    ptr[offset * 3 + 2] = (int)(b * 255);

}

void save_ppm(const char* filename, unsigned char* bitmap, int dim) {
    std::ofstream
    file(filename, std::ios::out | std::ios::binary);
    file << "P6\n" << dim << " " << dim << "\n255\n";
    for (int i = 0; i < dim * dim; i++) {
        file << bitmap[i * 3 + 0] << bitmap[i * 3 + 1] << bitmap[i * 3 + 2];
    }
    file.close();

}

int main() {
    unsigned char* dev_bitmap;
    cudaMalloc((void**)&dev_bitmap, DIM * DIM * 3);

    Sphere *temp_spheres = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_spheres[i].center.x = rnd(DIM *1.0f) - DIM / 2;
        temp_spheres[i].center.y = rnd(DIM *1.0f) - DIM / 2;
        temp_spheres[i].center.z = rnd(DIM *1.0f) - DIM / 2;
        temp_spheres[i].radius = rnd(DIM / 10.0f) + DIM / 50;
        temp_spheres[i].color.x = rnd(1.0f);
        temp_spheres[i].color.y = rnd(1.0f);
        temp_spheres[i].color.z = rnd(1.0f);
    }
    cudaMemcpyToSymbol(spheres, temp_spheres, sizeof(Sphere) * SPHERES); // Copy to device
    free(temp_spheres);

    dim3 grids(DIM / 16, DIM / 16);
    dim3 threads(16, 16);
    kernel<<<grids, threads>>>(dev_bitmap);

    unsigned char* bitmap = (unsigned char*)malloc(DIM * DIM * 3);
    cudaMemcpy(bitmap, dev_bitmap, DIM * DIM * 3, cudaMemcpyDeviceToHost);

    save_ppm("rayTrace.ppm", bitmap, DIM);

    free(bitmap);
    cudaFree(dev_bitmap);
    return 0;
}