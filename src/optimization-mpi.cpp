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

#include "interval.h"
#include "functions.h"
#include "minimizer.h"


#define NB_MINS 2


using namespace std;

bool idle_processors[50] = {false};

// TODO :
int find_idle_processors() {
	int idle_proc = -1;
	for (int i = 1; i < 50; i++) {
		if (idle_processors[i]) {
			idle_proc = i;
		}
	}
	return idle_proc;
}

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
	if (rank == 0) {
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
		for (int current_task = 0; current_task < 3; i++) {
			int next_idle_proc = find_idle_processors();
			if (next_idle_proc != -1) {
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
				int work = 1;
				// Send work to next rank
				s.str("");
				s << "Envoi du message WORK à " << next_idle_proc;
				echo(rank, s.str());
				MPI_Send(&work, 1, MPI_INT, next_idle_proc, 0, MPI_COMM_WORLD);

				// Send work to next rank
				MPI_Send(&to_send, 7, MPI_DOUBLE, next_idle_proc, 1, MPI_COMM_WORLD);
				s.str("");
				s << "Envoi des données à " << next_idle_proc << ": ";
				for(int l = 0; l < 7; l++) {
					s << to_send[l] << " ";
				}
				echo(rank, s.str());
				
				idle_processors[next_idle_proc] = false;
				
				// TODO : quand on reçoit la réponse, on remet le statut idle du processor à true
				
			} else {
				// If there is no idle processor, rank 0 will work
				minimize(f, p[current_task].first, p[current_task].second, threshold, min_ub, ml, rank, false);
				s.str("");
				s << "Min : " << min_ub;
				echo(rank, s.str());
			}
		}
		// Rank 0 works on the last box to split it again
		minimize(f, p[3].first, p[3].second, threshold, min_ub, ml, rank, false);
		
	} else {
		for (int i = 0; i < 4; i++) {
			minimize(f, p[i].first, p[i].second, threshold, min_ub, ml, rank, false);
		}
		echo(rank, "MESSAGE");
	}
}


int main(int argc, char** argv)
{
	// Prepare MPI
	int numprocs, rank, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Init(&argc, &argv);
	
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Get_processor_name(processor_name, &namelen);

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
		for (int i = 1; i < numprocs; i++) {
			idle_processors[i] = true;
		}
	
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
		
		// Find its own local minimum
		minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums, rank, true);
		mins.insert(min_ub);
		
		/*
		// Receive later other local minimums
		double other_min[NB_MINS];
		MPI_Status status;
		MPI_Request req;
		MPI_Irecv(&other_min, NB_MINS, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req);

		// Wait for other processors
		MPI_Barrier(MPI_COMM_WORLD);
		
		// Wait for response and add it to the minimums set
		MPI_Wait(&req, &status);
		for (int i = 0; i < NB_MINS; i++) {
			mins.insert(other_min[i]);
		}
		
		// Save the lowest min
		double lowest_min = *(mins.begin());
		
		// Displaying all potential minimizers
		copy(minimums.begin(),minimums.end(),
				 ostream_iterator<minimizer>(cout,"\n"));
		cout << "Number of minimizers: " << minimums.size() << endl;
		cout << "Upper bound for minimum: " << lowest_min << endl;
		*/
		int work = 0;
		for(int i = 1; i < numprocs; i++) {
			s.str("");
			s << "Abandon " << i;
			echo(rank, s.str());
			MPI_Send(&work, 1, MPI_INT, numprocs, 0, MPI_COMM_WORLD);
		}
		
	} else {
		// Receiving data to process
		double data[7] = {0.0};
		int work;
		//double mins[10];
		MPI_Status status;
		MPI_Request req1;
		MPI_Request req2;
		
		MPI_Irecv(&work, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
		MPI_Irecv(&data, 7, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req2);
		
		MPI_Wait(&req1, &status);
		while (work) {
			MPI_Wait(&req2, &status);
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
			
			echo(rank, "TRAITEMENT");
			//TODO : traitement
			//minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums, rank, false);
			
			MPI_Irecv(&work, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &req1);
			MPI_Irecv(&data, 7, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &req2);
			MPI_Wait(&req1, &status);
		}
		MPI_Cancel(&req2);
		echo(rank, "not work");
		

		/*
		// Rebuild global data
		precision = data[NB_MINS*4+1];
		min_ub = data[NB_MINS*4+2];
		
		// Build local data
		for(i = 0; i < NB_MINS; i++) {
			int var = 4*i;
			fun.x = interval(data[var+1],data[var+2]);
			fun.y = interval(data[var+3],data[var+4]);
			
			// Find minimum
			minimize(fun.f, fun.x, fun.y, precision, min_ub, minimums, rank, false);
			mins[i] = min_ub;
		}

		// Wait for other processors
		MPI_Barrier(MPI_COMM_WORLD);
		
		// Send local minimum found to rank 0
		MPI_Send(&mins, NB_MINS, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		*/
	}
	
	s.str("");
	s << "Upper bound for minimum: " << min_ub;
	echo(rank, s.str());
	
	MPI_Finalize();
}
