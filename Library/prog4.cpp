
/*

COMP 597
Allen Brubaker
ajb5377@psu.edu
5/5/2009
Problem 4

*/


#include <vector>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <string>

#include "Vector3.h"
#include "Vector2.h"
#include "Image.h"
#include "Image.cc"
#include "ONB.h"
#include "ONB.cc"
#include "Matrix.h"
#include "Matrix.cc"
#include "Ray.h"
#include "MarbleTexture.h"
#include "MarbleTexture.cc"
#include "SolidNoise.h"
#include "SolidNoise.cc"
#include "Texture.h"



const float FLOATMAX = 4294967295.0f;
const float pi = 3.14159265359f;

using namespace std;
		
		  

float randf()
{
	unsigned long seed = rand();
    unsigned long mult = 62089911UL;
    unsigned long llong_max = 4294967295UL;
    float float_max = 4294967295.0f;

    seed = mult * seed;
    return float(seed % llong_max) / float_max;  
}		
		
		
	
	
class SurfaceHit
{
	public:
	Vector3 hitPoint, normal;
	rgb kd, ks, ke;
	float surfaceArea;
	SurfaceHit() { }
	SurfaceHit(float _surfaceArea, Vector3 _hitPoint, Vector3 _normal, rgb _kd, rgb _ks, rgb _ke)
	{
		surfaceArea = _surfaceArea;
		hitPoint = _hitPoint; 
		normal = _normal; 
		kd = _kd;
		ks = _ks;  		
		ke = _ke;
	}
};
			
class Triangle 
{
	

    public:
	Vector3 p0, p1, p2; 
	Vector2 uv0, uv1, uv2;
	Image* texture;
	string colorOption;
	float surfaceArea;
	
	// Intersection vars after Hit() 
	Vector3 hitPoint, normal;
	double t, alpha, beta, gamma, u, v; 
	rgb kd, ks, ke;
	  
	Triangle() { }
	Triangle(Vector3 _p0, Vector3 _p1, Vector3 _p2,  Vector2 _uv0, Vector2 _uv1, Vector2 _uv2, string _colorOption, rgb _kd, rgb _ks, rgb _ke, Image* _texture = NULL) 
	{ 
		p0 = _p0; p1 = _p1; p2 = _p2; 
		uv0 = _uv0, uv1 = _uv1, uv2 = _uv2;  
		kd = _kd; 
		ks = _ks;
		ke = _ke;
		texture = _texture;
		colorOption = _colorOption;
		surfaceArea = cross(p1-p0, p2-p0).length()/2.0f;
	}
	
	// sets t, beta, gamma, color
	bool Hit(Ray r, float tMin=0, float tMax=FLOATMAX)
	{	
		double A = p0.e[0] - p1.e[0];
		double B = p0.e[1] - p1.e[1];
		double C = p0.e[2] - p1.e[2];
		
		double D = p0.e[0] - p2.e[0];
		double E = p0.e[1] - p2.e[1];
		double F = p0.e[2] - p2.e[2];
		
		double G = r.direction().x();
		double H = r.direction().y();
		double I = r.direction().z();
		
		double J = p0.x() - r.origin().x();
		double K = p0.y() - r.origin().y();
		double L = p0.z() - r.origin().z();
		
		double EIHF = E*I-H*F;
		double GFDI = G*F-D*I;
		double DHEG = D*H-E*G;
		
		double denom = (A*EIHF + B*GFDI + C*DHEG);
		
		beta = (J*EIHF + K*GFDI + L*DHEG) / denom;
		
		if (beta < 0.0 || beta > 1) 
			return false;
	
		double AKJB = A*K - J*B;
		double JCAL = J*C - A*L;
		double BLKC = B*L - K*C;
		
		gamma = (I*AKJB + H*JCAL + G*BLKC)/denom;
		if (gamma < 0.0 || beta + gamma > 1.0) 
			return false;
		
		t =  -(F*AKJB + E*JCAL + D*BLKC)/denom;
		if (t < tMin || t > tMax)
			return false;
		
			alpha = 1 - beta - gamma;
			hitPoint = alpha * p0 + beta * p1 + gamma * p2;	
			normal = unitVector(cross((p1 - p0), (p2 - p0)));
			u = alpha * uv0[0] + beta * uv1[0] + gamma * uv2[0];
			v = alpha * uv0[1] + beta * uv1[1] + gamma * uv2[1];
			if (u < 0) u = 0;
			if (u >= 1) u = .9999999999;
			if (v < 0) v = 0;
			if (v >= 1) v = .9999999999;
			return true;
	}		
	
	rgb GenerateColor()
	{
		if (colorOption == "texture")
		{
			double g, h, s, t;
			int nx, ny, x, y;
			
			nx = texture->width();
			ny = texture->height();
			
			v = .5 + v;
			if (v > .99999)
				v = 2-v;
			
			g = (nx-1) * u;
			h = (ny-1) * v;
			x = int(g);
			y = int(h);
			s = g - x;
			t = h - y;
			
			rgb c00 = texture->getPixel(x, y);
			rgb c01 = texture->getPixel(x, y+1);
			rgb c10 = texture->getPixel(x+1, y);
			rgb c11 = texture->getPixel(x+1, y+1);
			double A00 = (1-s)*(1-t);
			double A01 = (1-s)*(t);
			double A10 = (s)  *(1-t);
			double A11 = (s) * (t); 
			
			kd = c00 * A00 + c01 * A01 + c10 * A10 + c11 * A11; 
		}
		else if (colorOption == "marble")         
		{
			MarbleTexture marble(1, .50, 8);
			kd = marble.value(Vector2(), hitPoint);
		}
		// else colorOption == "none".. do nothing.. just leave default kd.  
		return kd;
	}	 
	
	SurfaceHit GetSurfaceHit()
	{
		return SurfaceHit(surfaceArea, hitPoint, normal, kd, ks, ke);
	}			  
	
	Vector3 HitPoint(float u, float v)
	{
		if (u+v > 1)
		{
			u = 1 - u;
			v = 1 - v;
		}
		return(p0 + u*(p1-p0) + v*(p2-p0));
	}            
};		



class Sphere
{
	

    public:
	Vector3 center;
	float radius;
	Image* texture;
	string colorOption;
	float surfaceArea;
	
	// Hit vars.
	Vector3 hitPoint, normal;
	double t, u, v;
	rgb kd, ks, ke;

	 
	Sphere() { }
	Sphere(Vector3 _center, float _radius, string _colorOption, rgb _kd, rgb _ks, rgb _ke, Image* _texture = NULL)
	{ 
		center = _center;
		radius = _radius;  
		texture = _texture;
		colorOption = _colorOption;
		kd = _kd; 
		ks = _ks;
		ke = _ke;
		surfaceArea = 4 * pi * radius*radius * 1/2;  // divide by 2 because we only care about surface area we can see.
	}
	
	bool Hit(Ray r, float tMin=0, float tMax=FLOATMAX)
	{	
		bool oneT;
       Vector3 temp = r.origin() - center;
       float twoa = 2.0f*dot( r.direction(), r.direction() );
       float b = 2.0f*dot( r.direction(), temp );
       float c = dot( temp, temp ) - radius*radius;
       float discriminant = b*b- 2.0f*twoa*c;   
       if (discriminant > 0.0f) 
       {                           
            discriminant = sqrt(discriminant);
            oneT = false;
			t = (-b - discriminant) / twoa;  
            if (t < tMin) 
			{                              
				oneT = true;
				t = (-b + discriminant) / twoa;
			}
            if (t < tMin || t > tMax) return false;
               
			// We have a hit!  Populate hit vars.
            hitPoint = r.pointAtParameter(t);
            if (!oneT) // two intersections, normal points outwards
				normal = unitVector(hitPoint - center);
			else // one intersection, normal points inwards
				normal = unitVector(center - hitPoint);
				
            float theta = acos( 0.9999f * (hitPoint.z()-center.z())/radius ); 
			float sinTheta = sin(theta);
            float phi = acos( normal.x() / (1.0001f * sinTheta * radius) );
            if (hitPoint.y() - center.y() < 0.0f) phi = 2*pi - phi;
            u = phi/(2.0f*pi);
		    v = 1.0f - theta/pi;
		    
			if (u < 0) u = 0;
			if (u >= 1) u = .9999999999;
			if (v < 0) v = 0;
			if (v >= 1) v = .9999999999;

            return true;
       }
       else  
          return false;
	}					
	rgb GenerateColor()
	{
		if (colorOption == "texture")
		{
			double g, h, s, t;
			int nx, ny, x, y;
			
			nx = texture->width();
			ny = texture->height();
			
			g = (nx-1) * u;
			h = (ny-1) * v;
			x = int(g);
			y = int(h);
			s = g - x;
			t = h - y;
			
			rgb c00 = texture->getPixel(x, y);
			rgb c01 = texture->getPixel(x, y+1);
			rgb c10 = texture->getPixel(x+1, y);
			rgb c11 = texture->getPixel(x+1, y+1);
			double A00 = (1-s)*(1-t);
			double A01 = (1-s)*(t);
			double A10 = (s)  *(1-t);
			double A11 = (s) * (t); 
			
			kd = c00 * A00 + c01 * A01 + c10 * A10 + c11 * A11; 
		}
		else if (colorOption == "marble")
		{
			MarbleTexture marble(1, .50, 8);
			kd = marble.value(Vector2(), hitPoint);
		}
		// else keep default kd
		return kd;
	}	
	
	SurfaceHit GetSurfaceHit()
	{
		return SurfaceHit(surfaceArea, hitPoint, normal, kd, ks, ke);
	}		
	
	Vector3 HitPoint(float u, float v)
	{
		Vector3 p;
		p[0] = radius*sin(pi*v)*cos(2*pi*u) + center.x();
		p[1] = radius*sin(pi*v)*sin(2*pi*u) + center.y();
		p[2] = radius*cos(pi*v) + center.z();
		return p;
	}		 			              
};		



class Cylinder
{
	

    public:
	Vector3 center;
	float radius;
	float height;
	Image* texture;
	string colorOption;
	float surfaceArea;
	
	// Hit vars.
	Vector3 hitPoint, normal;
	double t, u, v;
	rgb kd, ks, ke;
	  
	Cylinder() { }
	Cylinder(Vector3 _center, float _radius, float _height, string _colorOption, rgb _kd, rgb _ks, rgb _ke, Image* _texture = NULL)
	{ 
		center = _center;
		radius = _radius;  
		texture = _texture;
		height = _height;
		colorOption = _colorOption;
		kd = _kd; 
		ks = _ks;
		ke = _ke;
		surfaceArea = 2*radius * pi * height;
		
	}
	
	bool Hit(Ray r, float tMin=0, float tMax=FLOATMAX)
	{	
		
		float oneT;
		
		Ray xzRay(Vector3(r.origin().x(), 0, r.origin().z()), Vector3(r.direction().x(), 0, r.direction().z()));
		Vector3 xzCenter(center.x(), 0, center.z());
		Vector3 temp = xzRay.origin() - xzCenter;
		float twoa = 2.0f*dot( xzRay.direction(), xzRay.direction() );
		float b = 2.0f*dot( xzRay.direction(), temp );
		float c = dot( temp, temp ) - radius*radius;
		float discriminant = b*b- 2.0f*twoa*c; 
		
		if (discriminant > 0.0f) 
		{
			discriminant = sqrt(discriminant);
			oneT = false;
			t = (-b - discriminant) / twoa;               
			if (t < tMin || t > tMax || r.pointAtParameter(t).y() < center.y() || r.pointAtParameter(t).y() > center.y() + height)	
			{	
				t = (-b + discriminant) / twoa;
				oneT = true;
			}
			if (t < tMin || t > tMax || r.pointAtParameter(t).y() < center.y() || r.pointAtParameter(t).y() > center.y() + height) 
				return false;
			
			// We have a hit!  Populate hit vars.
			hitPoint = r.pointAtParameter(t);
			Vector3 changingCenter = center;
			changingCenter[1] = hitPoint[1];  //center changes along y axis of hitpoint. 
			if (!oneT) // two intersections, normal points outwards
				normal = unitVector(hitPoint - changingCenter);
			else // one intersection, normal point inwards.
				normal = unitVector(changingCenter - hitPoint);
				
			float theta = acos(0.9999f * (hitPoint.z() - center.z())/radius);
			//cout << theta << endl;
			if ((hitPoint.x() - center.x()) < 0)
				theta = 2*pi - theta;
			u = theta/(2*pi);
			v = (hitPoint.y() - center.y())/height;
			
			if (u < 0) u = 0;
			if (u >= 1) u = .9999999999;
			if (v < 0) v = 0;
			if (v >= 1) v = .9999999999;
			return true;
       }
       else
          return false;
	}		
	
	
	rgb GenerateColor()
	{
		if (colorOption == "texture")
		{
			double g, h, s, t;
			int nx, ny, x, y;
			
			nx = texture->width();
			ny = texture->height();
			
			g = (nx-1) * u;
			h = (ny-1) * v;
			x = int(g);
			y = int(h);
			s = g - x;
			t = h - y;
			
			rgb c00 = texture->getPixel(x, y);
			rgb c01 = texture->getPixel(x, y+1);
			rgb c10 = texture->getPixel(x+1, y);
			rgb c11 = texture->getPixel(x+1, y+1);
			double A00 = (1-s)*(1-t);
			double A01 = (1-s)*(t);
			double A10 = (s)  *(1-t);
			double A11 = (s) * (t); 
			
			kd = c00 * A00 + c01 * A01 + c10 * A10 + c11 * A11; 
		}
		else if (colorOption == "marble")
		{
			MarbleTexture marble(1, .50, 8);
			kd = marble.value(Vector2(), hitPoint);
		}
		// else keep default kd.
			
		return kd;	
	}	 	
	
	SurfaceHit GetSurfaceHit()
	{
		return SurfaceHit(surfaceArea, hitPoint, normal, kd, ks, ke);
	}			
	
	Vector3 HitPoint(float u, float v)
	{
		Vector3 p;
		p[2] = radius * cos(2*pi*u) + center.z();
		p[0] = radius * sin(2*pi*u) + center.x();
		p[1] = v * height + center.y();
	}		 
				 			              
};		





class RayTracer
{
	
	public:
		
	int nx, ny, sqrtNSamples;
	char *inputFile, *outputFile;
	
	vector<Triangle> triangles;
	vector<Sphere> spheres;
	vector<Cylinder> cylinders;
	rgb background, ambient;
	Vector3 lookfrom, lookat, vup; 	
	float vfov, d;
	
	ONB basis;
	Matrix M;
	Image image;
	Image textures[10000];
	int textureIndex;
	
	
	RayTracer(float _d, int _nx, int _ny, int _sqrtNSamples, char* _inputFile, char* _outputFile)
	{	
		d = _d;
		nx = _nx; 
		ny=_ny; 
		sqrtNSamples=_sqrtNSamples;
		inputFile = _inputFile; 
		outputFile = _outputFile;
		GetData();
	}


    void GetData()
	{
		string option;
		char triangleFile[40], triangleTextureFile[40], junk[40];
		char squareFile[40]; 
		char sphereFile[40], sphereTextureFile[40];
		char cylinderFile[40], cylinderTextureFile[40];
		ifstream in(inputFile);
		Vector3 p0, p1, p2, p3, center;
		Vector2 uv0, uv1, uv2;
		float radius, height;
		string colorOption;
		textureIndex = -1;
		rgb kd, ks, ke;

		

		while (in >> option)
		{
			if (option == "triangles")
			{
				in >> triangleFile >> colorOption; 
				cout << "triangles: " << triangleFile << " " << colorOption << endl;
				
				if (colorOption != "none" && colorOption != "marble")
				{
					strcpy(triangleTextureFile, colorOption.c_str());
					colorOption = "texture";
					++textureIndex;
					textures[textureIndex].readPPM(triangleTextureFile);
				}
				
				ifstream in2(triangleFile);
				while (in2 >> p0 >> uv0 >> p1 >> uv1 >> p2 >> uv2 >> kd >> ks >> ke)
				{	                   
					triangles.push_back(Triangle(p0, p1, p2, uv0, uv1, uv2, colorOption, kd, ks, ke, colorOption == "texture" ? &textures[textureIndex] : NULL));	
				}
				in2.close();
			}
			// A square is made up of two triangles, uv coordinates figured out automatically.
			if (option == "squares")
			{
				in >> squareFile >> colorOption; 
				cout << "squares: " << squareFile << " " << colorOption << endl;
				
				if (colorOption != "none" && colorOption != "marble")
				{
					strcpy(triangleTextureFile, colorOption.c_str());
					colorOption = "texture";
					++textureIndex;
					textures[textureIndex].readPPM(triangleTextureFile);
				}
				
				ifstream in5(squareFile);
				while (in5 >> p0 >> p1 >> p2 >> p3 >> kd >> ks >> ke)
				{	                   
					triangles.push_back(Triangle(p0, p1, p2, Vector2(0,0), Vector2(1,0), Vector2(1,1), colorOption, kd, ks, ke, colorOption == "texture" ? &textures[textureIndex] : NULL));	
					triangles.push_back(Triangle(p0, p2, p3, Vector2(0,0), Vector2(1,1), Vector2(0,1), colorOption, kd, ks, ke, colorOption == "texture" ? &textures[textureIndex] : NULL));
				}
				in5.close();
			}	
			if (option == "spheres")
			{
				in >> sphereFile >> colorOption;
				cout << "spheres: " << sphereFile << " " << colorOption << endl;
				
				if (colorOption != "none" && colorOption != "marble")
				{
					strcpy(sphereTextureFile, colorOption.c_str());
					colorOption = "texture";
					++textureIndex;
					textures[textureIndex].readPPM(sphereTextureFile);
				}
				
				ifstream in3(sphereFile);
				while (in3 >> center >> radius >> kd >> ks >> ke) 
				{
					spheres.push_back(Sphere(center, radius, colorOption, kd, ks, ke, colorOption == "texture" ? &textures[textureIndex] : NULL));
				}
				in3.close();
			}
			if (option == "cylinders")
			{
				in >> cylinderFile >> colorOption;
				cout << "cylinders: " << cylinderFile << " " << colorOption << endl;
				
				if (colorOption != "none" && colorOption != "marble")
				{
					strcpy(cylinderTextureFile, colorOption.c_str());
					colorOption = "texture";
					++textureIndex;	
					textures[textureIndex].readPPM(cylinderTextureFile);
				}
				ifstream in4(cylinderFile);
				while (in4 >> center >> radius >> height >> kd >> ks >> ke)
				{
					cylinders.push_back(Cylinder(center, radius, height, colorOption, kd, ks, ke, colorOption == "texture" ? &textures[textureIndex] : NULL));
				}
				in4.close();
			}
			if (option == "ambient") in >> ambient;
			if (option == "background") in >> background; 
			if (option == "lookfrom") in >> lookfrom; 
			if (option == "lookat") in >> lookat; 
			if (option == "vfov") in >> vfov; 
			if (option == "vup") in >> vup; 
		} 
		
		cout << "ambient: " << ambient << endl;
		cout << "background: " << background << endl;
		cout << "lookfrom: " << lookfrom << endl;
		cout << "lookat: " << lookat << endl;
		cout << "vfov: " << vfov << endl;
		cout << "vup: " << vup << endl; 
		
		in.close();
		
		basis.initFromWV(-1 * (lookat - lookfrom), vup);
		M = GetTransformMatrix();		
		
	}
	
	Matrix GetTransformMatrix() 
	{
		float h = 2 * d * tan(vfov/2.0 * pi / 180); // Height of frame.   ** Convert to radians from vfov given in degrees.
		float w = (float) nx/ny * h; // Width of frame.
		
		cout << "vfov: " << vfov << "     h: " << h << "    w: " << w << endl;
		
		Matrix M, M1, M2, M3, M4;
		
		M = M1 = M2 = M3 = M4 = identityMatrix();
				
		// Translate by M1
		M1.x[0][3] = -nx/2.0;
		M1.x[1][3] = -ny/2.0;
		M1.x[2][3] = -d;		
		
		// Scale by M2
		M2.x[0][0] = h/ny;
		M2.x[1][1] = h/ny;   
		
		// Rotate by M3
		M3.x[0][0] = basis.u().x();   M3.x[0][1] = basis.v().x();   M3.x[0][2] = basis.w().x();    
		M3.x[1][0] = basis.u().y();   M3.x[1][1] = basis.v().y();   M3.x[1][2] = basis.w().y();
		M3.x[2][0] = basis.u().z();   M3.x[2][1] = basis.v().z();   M3.x[2][2] = basis.w().z();
		
		// Translate by M4
		M4.x[0][3] = lookfrom.x();
		M4.x[1][3] = lookfrom.y();
		M4.x[2][3] = lookfrom.z();
	
		// Transform M
		M = M4 * M3 * M2 * M1;
		
		cout << endl << M << endl;
		
		return M;
		
	}
	
	
	Image RayTrace() 
	{ 	  
	 	Image image(nx, ny);  
	 	
	 	float xk, yk, a, b;
	 	rgb pixel;
	 	Vector3 p;
	 	Vector3 pp;
	 	Ray ray;
	 	float u, v;
		int pctComplete = -1;	  
			  
	 	float avg = 1.0/(sqrtNSamples*sqrtNSamples);  
	 	  
	  	for (int y=0; y<ny; ++y)
		{
			if (pctComplete < int((float) y/ny * 100))
			{	
				pctComplete = int((float) y/ny * 100);	
				if ((pctComplete)%10 == 0)
					cout << endl;
				cout << pctComplete << "% ";	
			} 
			for (int x=0; x<nx; ++x)
			{	
				pixel = rgb(0,0,0); 	  	
				for (int k=0; k<sqrtNSamples; ++k)
					for (int l=0; l<sqrtNSamples; ++l)	
					{			
						
						// Tent Filtering
						//a = randf();
						//b = randf();
						//xk = x + .5 + (a < .5 ? (-1 + sqrt(2*a))/2.0 : (1 - sqrt(2 - 2*a))/2.0);
						//yk = y + .5 + (b < .5 ? (-1 + sqrt(2*b))/2.0 : (1 - sqrt(2 - 2*b))/2.0);
						
						// Box Filtering
						u = (l+randf())/sqrtNSamples;
						v = (k+randf())/sqrtNSamples;
						
						if (u < 0.000001) u = 0.000001;
						if (u >= 1) u = .9999999999;
						if (v < 0.000001) v = 0.000001;
						if (v >= 1) v = .9999999999;
						
						xk = x + u;
						yk = y + v;
											
						p[0] = xk; p[1] = yk; p[2] = 0;
						pp = M * p;  // Transform p to new coordinate system.  
						
						ray.setOrigin(lookfrom);
						ray.setDirection(pp - lookfrom);
				  
					  	pixel += RayColor(ray, u, v, 2);
					}
				
				pixel *= avg;
				image.setPixel(x,y, pixel);
			}
		}
		
		//this.image = image;
		return image;
	}								
		

	rgb RayColor(Ray ray, float u, float v, int recursionDepthLeft)
	{
		SurfaceHit surfaceHit;
		Ray reflectRay;
		
		if (recursionDepthLeft <= 0)
			return rgb(1, 1, 1); 
		
		if (RayClosestHit(surfaceHit, ray, d))
		{
			rgb color;
			color = surfaceHit.ke + surfaceHit.kd * ambient;
			if (!Dark(surfaceHit.kd))
			{
				color += surfaceHit.kd * DirectLight(surfaceHit, u, v);	
			}	
					   
			if (!Dark(surfaceHit.ks))
			{
				reflectRay.setOrigin(surfaceHit.hitPoint);
				reflectRay.setDirection(reflect(ray.direction(), surfaceHit.normal));
				color += surfaceHit.ks * RayColor(reflectRay, u, v, recursionDepthLeft-1);
			}           
		    
			return color;
		}
		else
			return background;
	}
		
			
 
		

	bool RayClosestHit(SurfaceHit& surfaceHit, Ray ray, float tmin=0, float tmax=FLOATMAX)
	{
		int closestTriangle = -1;
		for (int i=0; i<triangles.size(); ++i)
		{
			// Keep track of closestTriangle to our eye that was hit.
			if (triangles[i].Hit(ray, tmin, tmax))
				if (closestTriangle == -1)
					closestTriangle = i;
				else if (triangles[i].t < triangles[closestTriangle].t)
					closestTriangle = i;
		}
		
		int closestSphere = -1;
		for (int i=0; i<spheres.size(); ++i)
		{
			// Keep track of closestsphere to our eye that was hit.
			if (spheres[i].Hit(ray, tmin, tmax))
				if (closestSphere == -1)
					closestSphere = i;
				else if (spheres[i].t < spheres[closestSphere].t)
					closestSphere = i;
		}
		
		int closestCylinder = -1;
		for (int i=0; i<cylinders.size(); ++i)
		{
			// Keep track of closestsphere to our eye that was hit.
			if (cylinders[i].Hit(ray, tmin, tmax))
				if (closestCylinder == -1)
					closestCylinder = i;
				else if (cylinders[i].t < cylinders[closestCylinder].t) 
					closestCylinder = i;
		}
		
			
		float ts[3];
		ts[0] = closestTriangle != -1 ? triangles[closestTriangle].t : -1;
		ts[1] = closestSphere != -1 ? spheres[closestSphere].t : -1;
		ts[2] = closestCylinder != -1 ? cylinders[closestCylinder].t : -1;  
					
		int closestShape = -1;		
		for (int i = 0; i<3; ++i)
		{
			if (ts[i] != -1)
				if (closestShape == -1)
					closestShape = i;
				else if (ts[i] < ts[closestShape])
					closestShape = i;
		}	 
		
		if (closestShape == -1)
		{
			return false;
		}
		else
		{
			if (closestShape == 0) 
			{	
				triangles[closestTriangle].GenerateColor();
				surfaceHit = triangles[closestTriangle].GetSurfaceHit();
			}
			else if (closestShape == 1)
			{	
				spheres[closestSphere].GenerateColor();
				surfaceHit = spheres[closestSphere].GetSurfaceHit();
			}
			else
			{	
				cylinders[closestCylinder].GenerateColor();
				surfaceHit = cylinders[closestCylinder].GetSurfaceHit();
			}
		}
		
		return true;		
	}						
	
	
	
	
	bool Dark(rgb color)
	{
		return (color.r() < .0000001 && color.g() < .0000001 && color.b() < .0000001);
	} 
	
	// Go through all luminaires (considered targets; not sources!).
	rgb DirectLight(const SurfaceHit& source, float u, float v)
	{	
		rgb light = rgb(0,0,0);
		Vector3 targetHitPoint, between;
		SurfaceHit target, trash;
		Ray lightRay;
		float length;
			
		// Triangle luminaires
		for (int i=0; i<triangles.size(); ++i)
		{
			if (!Dark(triangles[i].ke))
			{
				lightRay.setOrigin(source.hitPoint); 
				lightRay.setDirection(triangles[i].HitPoint(u,v)-source.hitPoint); 
				if (triangles[i].Hit(lightRay))
				{
					triangles[i].GenerateColor();
					target = triangles[i].GetSurfaceHit();
					lightRay.setDirection(target.hitPoint - source.hitPoint);
					if (!RayClosestHit(trash, lightRay, 0.00001, .99999))
					{
						between = source.hitPoint - target.hitPoint;
						light += 1/pi * cos_theta(lightRay.direction(), source.normal)*cos_theta(-lightRay.direction(), target.normal) * target.surfaceArea * target.ke *  1/between.squaredLength(); 
					}
				}
			}
		}
		// Sphere luminaires
		for (int i=0; i<spheres.size(); ++i)
		{
			if (!Dark(spheres[i].ke))
			{                  
				lightRay.setOrigin(source.hitPoint); 
				lightRay.setDirection(spheres[i].HitPoint(u,v)-source.hitPoint); 
				if (spheres[i].Hit(lightRay)) 
				{                         
					spheres[i].GenerateColor();
					target = spheres[i].GetSurfaceHit();
					// It automatically retrieves closest intersection point.  Update the ray with this info.
					lightRay.setDirection(target.hitPoint - source.hitPoint);
					if (!RayClosestHit(trash, lightRay, 0.00001, 0.99999))
					{
						between = source.hitPoint - target.hitPoint;  
						light += 1/pi * cos_theta(lightRay.direction(), source.normal)*cos_theta(-lightRay.direction(), target.normal) * target.surfaceArea * target.ke * 1/between.squaredLength(); 
					}              
				} 
			}
		}		
		// Cylinder Luminaires
		for (int i=0; i<cylinders.size(); ++i)
		{
			if (!Dark(cylinders[i].ke))
			{
				lightRay.setOrigin(source.hitPoint); 
				lightRay.setDirection(cylinders[i].HitPoint(u,v)-source.hitPoint); 
				if (cylinders[i].Hit(lightRay))
				{
					cylinders[i].GenerateColor();
					target = cylinders[i].GetSurfaceHit();
					lightRay.setDirection(target.hitPoint - source.hitPoint);
					if (!RayClosestHit(trash, lightRay, 0.00001, .99999))
					{
						between = source.hitPoint - target.hitPoint;
						light += 1/pi * cos_theta(lightRay.direction(), source.normal)*cos_theta(-lightRay.direction(), target.normal) * target.surfaceArea * target.ke / between.squaredLength(); 
					}
				}
			}
		}	
		return light;
	}		
				
						
		
};	

			
		   


int main(int argc, char *argv[])
{
	int nx = atoi(argv[1]);
	int ny = atoi(argv[2]); 
	int sqrtNSamples = atoi(argv[3]);
	char* inputFile = argv[4];
	char* outputFile = argv[5];
	
	
	RayTracer tracer(1.0, nx, ny, sqrtNSamples, inputFile, outputFile);
	
	cout << "Ray Tracing..." << endl;
	
	Image raster = tracer.RayTrace();
	
	cout << "Generated Image" << endl;
	cout << "Outputting to file" << endl;
	
	ofstream out(outputFile);
	raster.WritePPMASCII(out);	
	out.close();
	
	
}
	
	   
				  
										 
