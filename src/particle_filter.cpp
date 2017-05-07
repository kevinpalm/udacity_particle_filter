/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * 
 * Assignment TODOs completed by: Kevin Palm, 5/2017
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include <math.h>
#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	// set the number of particles
	num_particles = 1000;
	
	// unpack standard deviations for x, y, and theta for better readability
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];
	
	// generate gaussian distributions for each dimension
	std::normal_distribution<double> dist_x(x, std_x);
	std::normal_distribution<double> dist_y(y, std_y);
	std::normal_distribution<double> dist_theta(theta, std_theta);
	
	// lightweight random generator
	std::default_random_engine gen;
	
	// size of vectors for storing weights and particles
	weights.resize(num_particles);
	particles.resize(num_particles);
	
	// Generate Particles
	for (int i = 0; i < num_particles; ++i) {
		
		// declare the particle
		Particle p;
		
		// fill the features
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		
		// save a pointer to weights vector
		weights[i] = &p.weight;
		
		// save the particle
		particles[i] = p;

	}

	// init done
	is_initialized = true;
	std::cout << "Particles initialized" << std::endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// unpack standard deviations for x, y, and theta for better readability
	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];
	
	// lightweight random generator
	std::default_random_engine gen;
	
	// Update Particles
	for (int i = 0; i < num_particles; ++i) {

		// generate gaussian distributions for each dimension
		std::normal_distribution<double> dist_x(particles[i].x, std_x);
		std::normal_distribution<double> dist_y(particles[i].y, std_y);
		std::normal_distribution<double> dist_theta(particles[i].theta, std_theta);
		
		// check if zero yaw_rate
		if (yaw_rate != 0.0) {
			
			// update states for nonzero yaw_rate
			particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			particles[i].theta = fmod(particles[i].theta + yaw_rate * delta_t, 2 * M_PI);
			
		} else {
			
			// update states when zero yaw_rate
			particles[i].x = sin(particles[i].theta) * velocity * delta_t;
			particles[i].y = cos(particles[i].theta) * velocity * delta_t;
		}
		
		// add gaussian noise
		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
		
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
