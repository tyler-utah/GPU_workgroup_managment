/* requires:
   #include "lbabp.h"
 */

/*---------------------------------------------------------------------------*/

class Octree
{
	static const unsigned int MAXTREESIZE = 11000000;
	unsigned int numParticles;
	cl::Buffer particles;
	cl::Buffer newParticles;

	cl::Buffer tree;
	cl::Buffer treeSize;
	cl::Buffer particlesDone;
  cl::Buffer stealAttempts;

	float totalTime;

        /* stats */
        int maxMem;
        unsigned int* htree;
        unsigned int htreeSize;
        unsigned int hparticlesDone;
        unsigned int hstealAttempts;

public:

	void generateParticles(cl::CommandQueue queue);
	bool run(unsigned int threads, unsigned int blocks, int maxChildren, int numParticles);
	float printStats();
	float getTime();
	int getMaxMem();
};

/*---------------------------------------------------------------------------*/
