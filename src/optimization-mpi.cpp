/*
	Branch and bound algorithm to find the minimum of continuous binary 
	functions using interval arithmetic.

	MPI version

	From the work of Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
*/

#include <iostream>
#include <sstream>
#include <iterator>
#include <utility>
#include <string>
#include <stdexcept>
#include <array>
#include <mpi.h>
#include <unistd.h>
#include <omp.h>

#include "interval.h"
#include "functions.h"
#include "minimizer.h"


#define NB_MINS 2


using namespace std;

// Allow to print message with rank of the processor for debug purpose
void echo(const int rank, const string &message) {
	stringstream s;
	s << "Rang " << rank << " : " << message;
	cout << s.str() << endl;
}


// Split a 2D box into four subboxes by splitting each dimension
// into two equal subparts
void split_box(const interval& x, const interval& y,
				 interval &xl, interval& xr, interval& yl, interval& yr)
{
	double xm = x.mid();
	double ym = y.mid();
	xl = interval(x.left(),xm);
	xr = interval(xm,x.right());
	yl = interval(y.left(),ym);
	yr = interval(ym,y.right());
}


// Branch-and-bound minimization algorithm
void minimize(itvfun f,	// Function to minimize
				const interval& x, // Current bounds for 1st dimension
				const interval& y, // Current bounds for 2nd dimension
				double threshold,	// Threshold at which we should stop splitting
				double& min_ub,	// Current minimum upper bound
				minimizer_list& ml, // List of current minimizers
				const int rank, // Rank of the current processor
				const bool first_time) // Has Rank 0 to share work ?
{
	interval fxy = f(x,y);
	
	if (fxy.left() > min_ub) { // Current box cannot contain minimum?
		return ;
	}

	if (fxy.right() < min_ub) { // Current box contains a new minimum?
		min_ub = fxy.right();
		// Discarding all saved boxes whose minimum lower bound is 
		// greater than the new minimum upper bound
		auto discard_begin = ml.lower_bound(minimizer{0,0,min_ub,0});
		ml.erase(discard_begin,ml.end());
	}

	// Checking whether the input box is small enough to stop searching.
	// We can consider the width of one dimension only since a box
	// is always split equally along both dimensions
	if (x.width() <= threshold) { 
		// We have potentially a new minimizer
		ml.insert(minimizer{x,y,fxy.left(),fxy.right()});
		return ;
	}

	// The box is still large enough => we split it into 4 sub-boxes
	// and recursively explore them
	interval xl, xr, yl, yr;
	split_box(x,y,xl,xr,yl,yr);
	
	pair<interval, interval> p[4];
	p[0] = make_pair(xl,yl);
	p[1] = make_pair(xl,yr);
	p[2] = make_pair(xr,yl);
	p[3] = make_pair(xr,yr);

	stringstream s;

	// If conditions have been met, it splits the work 
	if ((rank == 0) && first_time) {
		int i = 0;
		double index = -1.0;
		for (auto x : functions) {
			if (x.second.f == f) {
				index = (double)i;
			} else {
				i++;
			}
		}
	
		// Allocate work for N processors
		//#pragma omp parallel for
		for (int current_task = 0; current_task < 4; current_task++) {
			if (current_task == 0) {
				// Rank 0 works on the last box to split it again
				minimize(f, p[0].first, p[0].second, threshold, min_ub, ml, rank, false);
			} else {
				// Send data to idle process
				double to_send[7] = {
					index,
					p[current_task].first.left(),
					p[current_task].first.right(),
					p[current_task].second.left(),
					p[current_task].second.right(),
					threshold,
					min_ub
				};
			
				// Send work to next rank
				MPI_Send(&to_send, 7, MPI_DOUBLE, current_task, 0, MPI_COMM_WORLD);
				s.str("");
				s << "Envoi des données à " << current_task << ": ";
				for(int l = 0; l < 7; l++) {
					s << to_send[l] << " ";
				}
				echo(rank, s.str());
			}

		}
	} else {
		//#pragma omp parallel for
		for (int i = 0; i < 4; i++) {
			minimize(f, p[i].first, p[i].second, threshold, min_ub, ml, rank, false);
		}
	}
}


int main(int argc, char** argv)
{
	// Prepare MPI
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	// MPI Initialization
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);


	// OMP Initialization
	omp_set_num_threads(4);

	// Variables declarations
	cout.precision(16);
	double min_ub = numeric_limits<double>::infinity();
	
	minimizer_list minimums;
	double precision;
	string choice_fun;
	opt_fun_t fun;
	bool good_choice;
	
	stringstream s;
	
	// Split work for processors
	if (rank == 0) {
		do {
			good_choice = true;

			cout << "Which function to optimize?\n";
			cout << "Possible choices: ";
			for (auto fname : functions) {
				cout << fname.first << " ";
			}
			cout << endl;
			cin >> choice_fun;
		
			try {
				fun = functions.at(choice_fun);
			} catch (out_of_range) {
				cerr << "Bad choice" << endl;
				good_choice = false;
			}
		} while(!good_choice);

		cout << "Precision? ";
		cin >> precision;
		
		set<double> mins;
		
		double minimum_received[3] = {0.0};
		MPI_Request reqs[3];
		MPI_Status status[3];
		
		for(int i = 1; i < 4; i++) {
			MPI_Irecv(&minimum_received[i-1], 7, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs[i-1]);
		}
		
		// Find its own local minimum
		minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums, rank, true);
		mins.insert(min_ub);
		
		for(int i = 0; i < 3; i++) {
			MPI_Wait(&reqs[i], &status[i]);
			mins.insert(minimum_received[i]);
		}
		
		s.str("");
		s << min_ub;
		echo(rank, s.str());
		
	  	/*
	  	for (set<double>::iterator it=mins.begin(); it!=mins.end(); ++it)
    		cout << ' ' << *it;
		*/
		
		min_ub = *(mins.begin());
	} else {
		// Receiving data to process
		double data[7] = {0.0};
		MPI_Status status;
		
		MPI_Recv(&data, 7, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		
		s.str("");
		for (int i = 0; i < 7; i++) {
			s << " " << data[i] << " ";
		}
		echo(rank, s.str());	
		
		// Find which function is involved
		int i = 0;
		int index = (int)data[0];
		for(auto x : functions) {
			if (i == index) {
				fun.f = x.second.f;
			} else {
				i++;
			}
		}
		
		fun.x = interval(data[1],data[2]);
		fun.y = interval(data[3],data[4]);
		
		precision = data[5];
		min_ub = data[6];
		
		minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums, rank, false);
		
		s.str("");
		s << min_ub;
		echo(rank, s.str());
		
		MPI_Send(&min_ub, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	
	s.str("");
	s << "Upper bound for minimum: " << min_ub;
	echo(rank, s.str());
	
	MPI_Finalize();
}
