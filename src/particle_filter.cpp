/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <float.h>

#include "particle_filter.h"


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
	num_particles = 50;

	// define gaussian distributions
	std::normal_distribution<double> dist_x(        x, std[0]);
	std::normal_distribution<double> dist_y(        y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	// Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS)
	for(int i = 0; i < num_particles; ++i) {
		Particle p;
		// Add random Gaussian noise to each particle.
		p.x      = dist_x(gen);
		p.y      = dist_y(gen);
		p.theta  = dist_theta(gen);

		p.id = i;

		// std::cout << "Particle i = " << i << " / x = " << p.x << " / y = " << p.y << " / O = " << p.theta << std::endl;

		particles.push_back(p);

		// set all weights to 1.
		p.weight = 1.0;
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double x_f, y_f, O_f;

	// update each particles position estimates and adding sensor noise
	for(int i = 0; i < num_particles; ++i) {

		if( yaw_rate != 0 ) {

			// The equations for updating x, y and the yaw angle when the yaw rate is not equal to zero:
			//   x_f = x_0 + v/O' * [sin(O + O'(dt)) - sin(O)]
			x_f = particles[i].x + velocity / yaw_rate * ( sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta) );

			//   y_f = y_0 + v/O' * [cos(O) - cos(O + O'(dt))]
			y_f = particles[i].y + velocity / yaw_rate * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t) );

			//   O_f = O + O'(dt)
			O_f = particles[i].theta + yaw_rate * delta_t;

		} else {

			// The equations for updating x, y and the yaw angle when the yaw rate is zero:
			//   x_f = x_0 + v(dt) * cos(O)
			x_f = particles[i].x + velocity * delta_t * cos(particles[i].theta);

			//   y_f = y_0 + v(dt) * sin(O)
			y_f = particles[i].y + velocity * delta_t * sin(particles[i].theta);

			//   O_f = O
			O_f = particles[i].theta;

		}

		// adding sensor noise to each measurement 	(mean .. particle position with std dev .. std dev of meas.)
		std::normal_distribution<double> dist_x(    x_f, std_pos[0]);
		std::normal_distribution<double> dist_y(    y_f, std_pos[1]);
		std::normal_distribution<double> dist_theta(O_f, std_pos[2]);

		particles[i].x     = dist_x(gen);
		particles[i].y     = dist_y(gen);
		particles[i].theta = dist_theta(gen);

	}

}

// return map<int,int> :: obs_id -> lm_id (i.e., observation-id -> closest landmark-id, if found - otherwise -1)
std::map<int,int> ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> lm_l, std::vector<LandmarkObs>& observations) {
	std::map<int,int> map_obs_to_lm;

	//std::cout << std::endl;
	//std::cout << " lm_l.size() " << lm_l.size() << " / observations.size() = " << observations.size() << std::endl;

	for(int m_obs = 0; m_obs < observations.size(); ++m_obs) {

			double optimal_dist = DBL_MAX;
			map_obs_to_lm[m_obs] = -1;

			for(int lm = 0; lm < lm_l.size(); ++lm) {

					double distance = dist( lm_l[lm].x_f, lm_l[lm].y_f, observations[m_obs].x, observations[m_obs].y );
					if( distance < optimal_dist ) {
						optimal_dist = distance;
						map_obs_to_lm[m_obs] = lm;
					}

			}

		}
		return map_obs_to_lm;

}

// transform between the VEHICLE'S coordinate system (where observations are given in) and
//  MAP'S coordinate system.
std::vector<LandmarkObs> ParticleFilter::transform_observation__vehicle_to_map_coord(const Particle& p, const std::vector<LandmarkObs>& observations) {
	std::vector<LandmarkObs> map_tranformed_observations;

	for(int j = 0; j < observations.size(); ++j) {
		double map_obs_x = p.x + observations[j].x * cos(p.theta) - observations[j].y * sin(p.theta);
		double map_obs_y = p.y + observations[j].x * sin(p.theta) + observations[j].y * cos(p.theta);
		int map_obs_id   = p.id;
		map_tranformed_observations.push_back(LandmarkObs(map_obs_id, map_obs_x, map_obs_y));
	}

	return map_tranformed_observations;
}

// based on the closest association (defined in the map), we calculate (normalized) weights for each particle using
//  a mult-variate Gaussian distribution
// return: weight
double ParticleFilter::update_weights_multv_gaussian_distr(
	const std::map<int,int>& map_obs_to_lm,
	const std::vector<Map::single_landmark_s>& lm_l,
	const std::vector<LandmarkObs>& obs,
	double std_landmark[] ) {

	double weight = 1.0;
	double div = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

	for (std::map<int,int>::const_iterator it = map_obs_to_lm.begin(); it != map_obs_to_lm.end(); ++it) {
			// std::cout << " closest_lm_id " << it->first << std::endl;
			if (it->first != -1) { // if there is even a closest association
				double diff_x = obs[it->first].x - lm_l[it->second].x_f;
				double diff_y = obs[it->first].y - lm_l[it->second].y_f;
				weight *= (1 / div) * exp( -0.5 * (pow(diff_x / std_landmark[0], 2) + pow(diff_y / std_landmark[1], 2) ) );
			}
	}
	return weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	for(int i = 0; i < num_particles; ++i) {

		// transform between the VEHICLE'S coordinate system (where observations are given in) and
		//  MAP'S coordinate system. (this transformation requires both rotation AND translation (but no scaling))
		std::vector<LandmarkObs> map_tranformed_observations = transform_observation__vehicle_to_map_coord(particles[i], observations);

		// predict measurements to all map landmarks within sensor range for each particle and
		//  associate sensor measurements to map landmarks.
		std::map<int,int> map_obs_to_lm = dataAssociation(map_landmarks.landmark_list, map_tranformed_observations);

		// based on association, we calculate (normalized) weights for each particle using
		//  a mult-variate Gaussian distribution
		weights[i] = update_weights_multv_gaussian_distr(map_obs_to_lm, map_landmarks.landmark_list, map_tranformed_observations, std_landmark); // particles[i].weight;

	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	//   using std::discrete_distribution

	for(int i = 0; i < weights.size(); ++i) {
		//std::cout << " weights[ " << i << "] = " << weights[i] << std::endl;
	}

	std::discrete_distribution<double> weight_distr(weights.begin(), weights.end());
	std::vector<Particle> new_particles;

	for (int i = 0; i < num_particles; ++i) {
		int index = weight_distr(gen);
		// std::cout << " -- picking index " << index << std::endl;
		Particle p = particles[index];
		new_particles.push_back(p);
	}

	particles = new_particles;
	// std::cout << std::endl;

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
