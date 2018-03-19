#include <pcl/PCLPointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/auto_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/surface/poisson.h>

#include <boost/make_shared.hpp>

using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

int default_depth = 8;
int default_solver_divide = 8;
int default_iso_divide = 8;
float default_point_weight = 4.0f;

void
compute (const pcl::PointCloud<PointXYZRGBNormal>::ConstPtr &input, PolygonMesh &output,
         int depth, int solver_divide, int iso_divide, float point_weight)
{
  PointCloud<PointXYZRGBNormal>::Ptr xyz_cloud (new pcl::PointCloud<PointXYZRGBNormal> ());
  //fromPCLPointCloud2 (*input, *xyz_cloud);

  print_info ("Using parameters: depth %d, solverDivide %d, isoDivide %d\n", depth, solver_divide, iso_divide);

	Poisson<PointXYZRGBNormal> poisson;
	poisson.setDepth (depth);
	poisson.setSolverDivide (solver_divide);
	poisson.setIsoDivide (iso_divide);
	poisson.setPointWeight (point_weight);
    poisson.setInputCloud (input);

  TicToc tt;
  tt.tic ();
  print_highlight ("Computing ...");
  poisson.reconstruct (output);

  print_info ("[Done, "); print_value ("%g", tt.toc ()); print_info (" ms]\n");
}

void
saveCloud (const std::string &filename, const PolygonMesh &output)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());
  pcl::io::savePLYFile(filename, output);

  print_info ("[done, "); print_value ("%g", tt.toc ()); print_info (" ms]\n");
}

int
main ()
{
  int depth = default_depth;
  print_info ("Using a depth of: "); print_value ("%d\n", depth);

  int solver_divide = default_solver_divide;
  print_info ("Setting solver_divide to: "); print_value ("%d\n", solver_divide);

  int iso_divide = default_iso_divide;
  print_info ("Setting iso_divide to: "); print_value ("%d\n", iso_divide);

  float point_weight = default_point_weight;
  print_info ("Setting point_weight to: "); print_value ("%f\n", point_weight);

  // Load the first file
  pcl::PointCloud<PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<PointXYZRGBNormal>());
  if (pcl::io::load("/home/shwetha/btp/video/ehd_demo.nvm.cmvs/04/models/option-0000.ply", *cloud) < 0)
  	{
      PCL_ERROR ("Unable to load.\n");
      return (-1);
	}

  // Apply the Poisson surface reconstruction algorithm
  PolygonMesh output;
  compute (cloud, output, depth, solver_divide, iso_divide, point_weight);

  // Save into the second file
  saveCloud ("/home/shwetha/btp/pcl_recon/ehd_demo_pcl_recon_output.ply", output);
}
