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


int number_idle_processors;
int next_idle_processor_to_call;

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
		
		if (check_idle_processors())
		
		/*
		// If there are idle processors, share with them 
		while (number_idle_processors != 0) {
			int number_tasks[4] = {0};
			
			// Number of processors ready to compute data
			int max_proc = 4;
			int working_proc = (number_idle_processors % max_proc) + 1;
			int working_proc_copy = working_proc;
			int current_task = 0;
			
			// Allocate work for N processors
			for (int i = 0; i < max_proc; i++) {
				number_tasks[i] = max_proc / working_proc;
				max_proc = max_proc - number_tasks[i];
				working_proc--;
				
				// For each task of a processor, share data
				for (int nb_task = 0; nb_task < number_tasks[i]; nb_task++) {
					
					if (i == working_proc - 1) {
						// Rank 0 work
						minimize(f, p[current_task].first, p[current_task].second, threshold, min_ub, ml, rank, false);
					} else {
						// For other ranks
						double to_send[7] = {
							index,
							p[current_task].first.left(),
							p[current_task].first.right(),
							p[current_task].second.left(),
							p[current_task].second.right(),
							threshold,
							min_ub
						};
						
						if(nb_task == 0) {
							// Send work to next rank
							s.str("");
							s << "Envoi du message '" << number_tasks[i] << "' NUMTASKS à " << next_idle_processor_to_call;
							echo(rank, s.str());
							MPI_Send(&number_tasks[i], 1, MPI_INT, next_idle_processor_to_call, 0, MPI_COMM_WORLD);
													
							// Wait for other processors
							MPI_Barrier(MPI_COMM_WORLD);
						}

						// Send work to next rank
						MPI_Send(&to_send, 7, MPI_DOUBLE, next_idle_processor_to_call, 0, MPI_COMM_WORLD);
						s.str("");
						s << "Envoi des données à " << next_idle_processor_to_call << ": ";
						for(int l = 0; l < 7; l++) {
							s << to_send[l] << " ";
						}
						echo(rank, s.str());
					}
					current_task++;
				}

			}
			*/
			
			/*
			next_idle_processor_to_call++;
			s.str("");
			s << "NB next idle proc to call : " << next_idle_processor_to_call << ", Nb idle proc : " << number_idle_processors;
			echo(rank, s.str());
			sleep(1);
			
			number_idle_processors = number_idle_processors - working_proc_copy + 1;
			s.str("");
			s << "NB next idle proc to call : " << next_idle_processor_to_call << ", Nb idle proc : " << number_idle_processors;
			echo(rank, s.str());
			*/
			
			/*
			// Fill an array of doubles to send it
			int size = 4 * NB_MINS + 3;
			double to_send[50] = {
				index,
				xl.left(),
				xl.right(),
				yl.left(),
				yl.right(),
				xl.left(),
				xl.right(),
				yr.left(),
				yr.right(),
				threshold,
				min_ub
			};
			
			// Send work to rank 1
			MPI_Send(to_send, size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
			
			// Half of the work for rank 0
			minimize(f, xr, yl, threshold, min_ub, ml, rank, false);
			minimize(f, xr, yr, threshold, min_ub, ml, rank, false);
			*/
	

		}
		minimize(f, xl, yl, threshold, min_ub, ml, rank, false);
		minimize(f, xl, yr, threshold ,min_ub, ml, rank, false);
		minimize(f, xr, yl, threshold, min_ub, ml, rank, false);
		minimize(f, xr, yr, threshold, min_ub, ml, rank, false);
	} else {
		minimize(f, xl, yl, threshold, min_ub, ml, rank, false);
		minimize(f, xl, yr, threshold ,min_ub, ml, rank, false);
		minimize(f, xr, yl, threshold, min_ub, ml, rank, false);
		minimize(f, xr, yr, threshold, min_ub, ml, rank, false);
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

	number_idle_processors = numprocs - 1;
	next_idle_processor_to_call = 1;

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
		
	} else {
		// Receiving data to process
		double data[4][10] = {{0.0}};
		//double mins[10];
		MPI_Request requests[4];
		MPI_Status status;
		
		int number_tasks = 0;
		MPI_Recv(&number_tasks, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		s.str("");
		s << "Réception de " << number_tasks << " blocs";
		echo(rank, s.str());
		
		// Wait for other processors
		MPI_Barrier(MPI_COMM_WORLD);
		
		for (int i = 0; i < number_tasks; i++) {
			MPI_Irecv(&data[i], 7, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[i]);

		}
		
		for (int i = 0; i < number_tasks; i++) {
			MPI_Wait(&requests[i], &status);
			s.str("");
			for (int j = 0; j < 7; j++) {
				s << data[i][j] << " ";
			}
			echo(rank, s.str());
		}
		
		/*
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
