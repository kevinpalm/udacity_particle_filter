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
#include "map.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	// set the number of particles
	num_particles = 5;
	
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
	x_vals.resize(num_particles);
	y_vals.resize(num_particles);
	
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
		
		// save to weights vector
		weights[i] = p.weight;
		
		// save to x and y coordinates
		x_vals[i] = p.x;
		y_vals[i] = p.y;
		
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
	
	// iterate through each observation
	for (int o = 0; o < observations.size(); o++) {
		
		// placeholder list for elucidian distances
		std::vector<double> distances(predicted.size());
		
		// iterate through each prediction
		for (int p = 0; p < predicted.size(); p++) {
			distances[p] = sqrt((predicted[p].x - observations[o].x) * (predicted[p].x - observations[o].x) + (predicted[p].y - observations[o].y) * (predicted[p].y - observations[o].y));
		}
		
		// find the index of the minimum distance
		int min_index = std::min_element(distances.begin(), distances.end()) - distances.begin();
		
		// set the associated ID
		observations[o].id = predicted[min_index].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	
	// Determine our region of interest
	auto x_ends = std::minmax_element(x_vals.begin(), x_vals.end());
	double x_min = *x_ends.first - sensor_range;
	double x_max = *x_ends.second + sensor_range;
	auto y_ends = std::minmax_element(y_vals.begin(), y_vals.end());
	double y_min = *y_ends.first - sensor_range;
	double y_max = *y_ends.second + sensor_range;
	
	// Create a smaller map which only includes the region of interest
	Map sliced_map;
	for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
		if ((map_landmarks.landmark_list[i].x_f >= x_min) &&
		(map_landmarks.landmark_list[i].x_f <= x_max) &&
		(map_landmarks.landmark_list[i].y_f >= y_min) &&
		(map_landmarks.landmark_list[i].y_f <= y_max)) {
			sliced_map.landmark_list.push_back(map_landmarks.landmark_list[i]);
		}
	}

	// helpers for multivariate Gaussian distribution later
	double x_std = std_landmark[0];
	double y_std = std_landmark[0];
	double x_var = sqrt(x_std);
	double y_var = sqrt(y_std);
	
	// iterate through the vector of particles
	for (int p = 0; p < num_particles; ++p) {
		
		// pull out the predicted landmarks that are relevent to this particle
		std::vector<LandmarkObs> predicted_observations;
		for (int i = 0; i < sliced_map.landmark_list.size(); ++i) {
			if ((sliced_map.landmark_list[i].x_f >= particles[p].x - sensor_range) &&
				(sliced_map.landmark_list[i].x_f <= particles[p].x + sensor_range) &&
				(sliced_map.landmark_list[i].y_f >= particles[p].y - sensor_range) &&
				(sliced_map.landmark_list[i].y_f <= particles[p].y + sensor_range)) {
					LandmarkObs lndmrk;
					lndmrk.x = sliced_map.landmark_list[i].x_f;
					lndmrk.y = sliced_map.landmark_list[i].y_f;
					lndmrk.id = i;
					predicted_observations.push_back(lndmrk);
			}
		}
		
		// convert the observations to the map coordinate system
		std::vector<LandmarkObs> mapped_observations = observations;
		for (int i = 0; i < observations.size(); i++) {
			mapped_observations[i].x = observations[i].x * cos(particles[p].theta) + observations[i].y * sin(particles[p].theta) + particles[p].x;
			mapped_observations[i].y = observations[i].x * sin(particles[p].theta) + observations[i].y * cos(particles[p].theta) + particles[p].y;
		}
		
		// make sure we have landmarks in our sensor range
		if (predicted_observations.size() > 0) {
			
			// associate the converted observations each to their nearest landmark
			dataAssociation(predicted_observations, mapped_observations);
			
			// placeholder for this particle's weight
			double particle_probability = 1.0;
			
			// compute multivariate Gaussian probability
			for (int i = 0; i < mapped_observations.size(); i++) {
				double x_diff = mapped_observations[i].x - predicted_observations[mapped_observations[i].id].x;
				double y_diff = mapped_observations[i].y - predicted_observations[mapped_observations[i].id].y;
				particle_probability *= (1/(2*M_PI*x_var*y_var))*exp(-((x_diff*x_diff)/(2*x_std) + (y_diff*y_diff)/(2*y_std)));
			}
			
			// save the result
			particles[p].weight = particle_probability;
			weights[particles[p].id] = particle_probability;
			
		} else {
			std::cout << "Error - a particle has drifted off the map." << std::endl;
		}	
	}
}

void ParticleFilter::resample() {

	// Placeholders for new particles
	std::vector<double> new_x_vals;
	std::vector<double> new_y_vals;
	std::vector<Particle> new_particles;
	
	// Define generator for weighted sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
	
	// Do the sampling
	for (int i=0; i<num_particles; i++) {

		// Choose an index
		int chosen = d(gen);
		new_particles.push_back(particles[chosen]);
		new_x_vals.push_back(particles[chosen].x);
		new_y_vals.push_back(particles[chosen].y);
	}
	
	std::cout << "Old Particles:" << std::endl;
	for (int x = 0; x < x_vals.size(); x++) {
		std::cout << x_vals[x];
		std:: cout << ", ";
	}
	std::cout << "." << std::endl;
		for (int x = 0; x < y_vals.size(); x++) {
		std::cout << y_vals[x];
		std:: cout << ", ";
	}
	std::cout << "." << std::endl;
	std::cout << "Weights:" << std::endl;
	for (int x = 0; x < weights.size(); x++) {
		std::cout << weights[x];
		std:: cout << ", ";
	}
	std::cout << "." << std::endl;
	
	// Save the new samples
	particles = new_particles;
	x_vals = new_x_vals;
	y_vals = new_y_vals;

	std::cout << "New Particles:" << std::endl;
	for (int x = 0; x < x_vals.size(); x++) {
		std::cout << x_vals[x];
		std:: cout << ", ";
	}
	std::cout << "." << std::endl;
		for (int x = 0; x < y_vals.size(); x++) {
		std::cout << y_vals[x];
		std:: cout << ", ";
	}
	std::cout << "." << std::endl;
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
