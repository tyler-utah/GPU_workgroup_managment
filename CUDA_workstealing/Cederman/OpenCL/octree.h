/* requires:
   #include "lbabp.h"
 */

/*---------------------------------------------------------------------------*/

enum LBMethod
{
	Dynamic,
	Static
};

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

	LBABP lbws;
        /* Hugues: we do not implement the static method in OpenCL */
	/* LBStatic lbstat; */

	float totalTime;
	LBMethod method;

        /* stats */
        int maxMem;
        unsigned int* htree;
        unsigned int htreeSize;
        unsigned int hparticlesDone;

public:

	void generateParticles(cl::CommandQueue queue);
	bool run(unsigned int threads, unsigned int blocks, LBMethod method, int maxChildren, int numParticles);
	float printStats();
	float getTime();
	int getMaxMem();
};

/*---------------------------------------------------------------------------*/
