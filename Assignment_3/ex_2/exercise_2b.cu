#include <iostream>
#include <string>
#include <stdexcept>
#include <random>
#include <stdio.h>
#include <sys/time.h>
#include <cmath>

#define NUM_ARGS 2

#define add_particles(a,b) make_float3(a.x+b.x, a.y+b.y, a.z+b.z)
#define mul_particles(a,b) make_float3(a.x*b.x, a.y*b.y, a.z*b.z)

using std::stoi;
using std::cout;
using std::endl;
using std::exception;
using std::abs;

struct Particle {
	float3 pos;
	float3 vel;

	Particle() {
		pos = make_float3(0,0,0);
		vel = make_float3(0,0,0);
	}

	Particle(float3 p, float3 v) : pos(p), vel(v) {}
};

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void particle_update_gpu(Particle *particles, const int num_particles) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < num_particles) {
		if(particles[i].pos.z > 0) {
			particles[i].vel = add_particles(particles[i].vel, make_float3(0,0,-9.82));
		} else {
			particles[i].vel = add_particles(particles[i].vel, make_float3(0,0,9.82));
		}
		particles[i].vel = mul_particles(particles[i].vel, make_float3(0.99,0.99,0.99));
		particles[i].pos = add_particles(particles[i].pos, particles[i].vel);
	}
}

__host__ void particle_update_cpu(Particle *particles, const int num_particles) {
	for(int i = 0; i < num_particles; i++) {
	    if(particles[i].pos.z > 0) {
	        particles[i].vel = add_particles(particles[i].vel, make_float3(0,0,-9.82));
	    } else {
	        particles[i].vel = add_particles(particles[i].vel, make_float3(0,0,9.82));
	    }
	    particles[i].vel = mul_particles(particles[i].vel, make_float3(0.99,0.99,0.99));
	    particles[i].pos = add_particles(particles[i].pos, particles[i].vel);
	}
}

bool compare(Particle a, Particle b) {
	float eps = 1e-6;
	if(abs(a.pos.x-b.pos.x) > eps) return false;
	if(abs(a.pos.y-b.pos.y) > eps) return false;
	if(abs(a.pos.z-b.pos.z) > eps) return false;
	if(abs(a.vel.z-b.vel.z) > eps) return false;
	if(abs(a.vel.z-b.vel.z) > eps) return false;
	if(abs(a.vel.z-b.vel.z) > eps) return false;
	return true;
}

bool compare(Particle *a, Particle *b, const int num_particles) {
	for(int i = 0; i < num_particles; i++) {
		if(compare(a[i],b[i]) == false) {
			cout << a[i].pos.x << " " << b[i].pos.x << endl;
			return false;
		}
	}
	return true;
}

int main(int argc, char **argv) {

	int num_particles;
	int num_iterations;
	int TPB;

	/* Read command line arguments */
		if(argc < NUM_ARGS + 1) {
			cout << "Too few arguments" << endl;
			return 0;
		}
		else {
			try {
				num_particles = stoi(argv[1]);
				num_iterations = stoi(argv[2]);
				TPB = stoi(argv[3]);
			} catch(exception e) {
                cout << "Invalid arguments. Proper usage is an integer number of particles and an integer number of iterations and an integer threads per block" << endl;
			}
		}
        
        /* INITIALISE RNG */
		std::random_device rd;
		std::mt19937 engine(rd());
		std::uniform_real_distribution<float> dist(-1000,1000);
        
        /* MEMORY ALLOCATION AND INITIALISATION */
		Particle *particles_cpu = new Particle[num_particles];
		Particle *particles_gpu;
        
        cudaMallocManaged(&particles_gpu, sizeof(Particle)*num_particles);

		float x, y, z;
		float vx, vy, vz;
		for(int i = 0; i < num_particles; i++) {
			x = dist(engine);
			y = dist(engine);
			z = dist(engine);
			vx = dist(engine);
			vy = dist(engine);
			vz = dist(engine);
			
			particles_cpu[i] = Particle(make_float3(x,y,z), make_float3(vx,vy,vz));
			particles_gpu[i] = Particle(make_float3(x,y,z), make_float3(vx,vy,vz));
		}

	/* EXECUTION */
		double time;

		time = cpuSecond();
			for(int i = 0; i < num_iterations; i++) {
				particle_update_cpu(particles_cpu, num_particles);
			}
		time = cpuSecond() - time;
		cout << "CPU TOOK " << time << " SECONDS" << endl;

		time = cpuSecond();          
            for(int i = 0; i < num_iterations; i++) {
                particle_update_gpu<<<(num_particles+TPB-1)/TPB, TPB>>>(particles_gpu, num_particles);
                cudaDeviceSynchronize();
			}
		time = cpuSecond() - time;
		cout << "GPU TOOK " << time << " SECONDS" << endl;
		if(compare(particles_cpu, particles_gpu, num_particles)) {
			cout << "COMPARISON SUCCESSFUL" << endl;
		} else {
			cout << "COMPARISON FAILED" << endl;
		}
	return 0;
}
