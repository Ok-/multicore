/*
	Branch and bound algorithm to find the minimum of continuous binary 
	functions using interval arithmetic.

	MPI version

	From the work of Frederic Goualard <Frederic.Goualard@univ-nantes.fr>
*/

#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <mpi.h>

#include "interval.h"
#include "functions.h"
#include "minimizer.h"

using namespace std;


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

	// If conditions have been met, it splits the work 
	if (first_time && (rank == 0))	{
		// TODO SEND INFO TO RANK 1
		minimize(f, xl, yl, threshold, min_ub, ml, rank, false);
		minimize(f, xl, yr, threshold, min_ub, ml, rank, false);
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

	// Variables declarations
	cout.precision(16);
	double min_ub = numeric_limits<double>::infinity();
	
	minimizer_list minimums;
	double precision;
	string choice_fun;
	opt_fun_t fun;
	bool good_choice;
	
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
		
		minimize(fun.f,fun.x,fun.y,precision,min_ub,minimums, rank, true);
	} else {
		cout << "nothing to do" << endl;
		// MPI_Recv(&msg)
	}
	
	// Displaying all potential minimizers
	copy(minimums.begin(),minimums.end(),
			 ostream_iterator<minimizer>(cout,"\n"));
	cout << "Number of minimizers: " << minimums.size() << endl;
	cout << "Upper bound for minimum: " << min_ub << endl;
	
	MPI_Finalize();
}
