
/*
CMPSC 457
Allen Brubaker
ajb5377@psu.edu
4/25/2010
Problem 4 - Raytraces a scene with diffuse and specular surfaces along with lights/shadows using spheres and triangles.  
*/

#include <cmath>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cfloat>
#include <vector>
#include <string>

using namespace std;

const float INFINITY = 4294967295.0f;
const float PI = 3.14159265359f;

double randf()
{
	unsigned long seed = rand();
    unsigned long mult = 62089911UL;
    unsigned long llong_max = 4294967295UL;
    float float_max = 4294967295.0f;

    seed = mult * seed;
    return double(seed % llong_max) / float_max;  
}	

class Vector
{
public:
  double x, y, z;

  Vector(double x=0, double y=0, double z=0)
    : x(x), y(y), z(z) {}

  Vector operator+ (const Vector& v) const
  { return Vector(x+v.x, y+v.y, z+v.z); }

  Vector operator- (const Vector& v) const
  { return Vector(x-v.x, y-v.y, z-v.z); }

  Vector operator-() const
  {
	  return Vector(-x, -y, -z);
  }
  
  Vector operator* (double d) const
  { return Vector(x*d, y*d, z*d); }

  Vector operator/ (double d) const
  {  return Vector(x/d, y/d, z/d);}

  Vector operator* (const Vector& v) const
  { return Vector(x*v.x, y*v.y, z*v.z); }

  // normalize this vector
  Vector& norm() 
  { return *this = *this * (1 / sqrt(x*x + y*y + z*z)); }

  // dot product
  double dot(const Vector& v) const
  { return x*v.x + y*v.y + z*v.z; }

  // cross product
  Vector cross(const Vector& v) const
  { return Vector(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x); }

  bool operator==(const Vector& v) const
  {
	  return abs(x - v.x) < 1.e-6 && abs(y-v.y) < 1.e-6 && abs(z-v.z) < 1.e-6;
  }
  bool operator!=(const Vector& v) const
  {
	  return !(*this == v);
  }


 Vector& operator+=(const Vector &v){
    x  += v.x;
    y  += v.y;
    z  += v.z;
    return *this;
}

 Vector& operator-=(const Vector& v) {
    x  -= v.x;
    y  -= v.y;
    z  -= v.z;
    return *this;
}

 Vector& operator*=(const float t) {
    x  *= t;
    y  *= t;
    z  *= t;
    return *this;
}

 Vector& operator/=(const float t) {
    x  /= t;
    y  /= t;
    z  /= t;
    return *this;
}



  double length() 
  { return sqrt(this->dot(*this)); } 

  Vector reflect(Vector norm) const
  {
	  Vector n = norm.norm();
	  Vector l = (-(*this)).norm();
	  return n * 2*n.dot(l) - l;
  }
};

inline Vector operator*(double t, const Vector &v) {
    return Vector(t*v.x, t*v.y, t*v.z); 
}

istream &operator>>(istream &is, Vector &t) {
   is >> t.x >> t.y >> t.z;
   return is;
}

ostream &operator<<(ostream &os, const Vector &t) {
   os << t.x << " " << t.y << " " << t.z;
   return os;
}



// Not the best way but it works
typedef Vector Point;
typedef Vector Color;


class Ray
{
public:
  Point origin;   // origin of the ray
  Vector dir;  // direction of the ray
  Ray(){}
  Ray (Point origin, Vector dir) : origin(origin), dir(dir.norm()) {} 
  Point PointAtDistance(double dist)
  {
	  return origin + dir * dist;
  }
};



class Image
{
public:
	int Height, Width;
	string FileName;
	Color** Canvas;
	Image(){}
	Image(int _width, int _height)
	{
		FileName = "";
		Height = _height; Width = _width; 
		
		Canvas = new Color*[Height];
		for (int i= 0; i<Height; ++i)
			Canvas[i] = new Color[Width];
	}

	// Canvas[0][0] is the bottom left of the image!
	static Image* FromFile(string fileName)
	{
		int width, height;
		Color c;
		ifstream in(fileName.c_str());
		string buffer;
		in >> buffer >> width >> height >> buffer; // ignore p6 and 255
		Image *image = new Image(width, height);
		image->FileName = fileName;
		cout << "Loading " << fileName << "....";
		for (int i=height-1; i>=0; --i)
			for (int j=0; j<width; ++j)
			{	
				in >> c; 
				image->Canvas[i][j] = c / 255.0;
			}
			
		in.close();
		cout << "Done" << endl;
		return image;
	}

	static Image* SearchImages(string textureName, vector<Image*>& textures)
	{
		for (unsigned int i=0; i<textures.size(); ++i)
			if (textures[i]->FileName == textureName)
				return textures[i];

		Image* image = FromFile(textureName);
		textures.push_back(image);
		return image;
	}

	// Canvas[0][0] is the bottom left of the image!
	void Output(string fileName)
	{
		ofstream out(fileName.c_str());

		out << "P3" << endl;
		out << Width << " " << Height << endl;
		out << "255" << endl;
		for (int i=Height-1; i>=0; --i)
		{
			for (int j=0; j<Width; ++j)
			{
				int R = int(Canvas[i][j].x * 255 + .5);
				int G = int(Canvas[i][j].y * 255 + .5);
				int B = int(Canvas[i][j].z * 255 + .5);
				R = R < 0 ? 0 : (R > 255 ? 255 : R);
				G = G < 0 ? 0 : (G > 255 ? 255 : G);
				B = B < 0 ? 0 : (B > 255 ? 255 : B);
				out << R << " " << G << " " << B << " ";
			}
			out << endl;
		}
		out.close();
	}
};




class ISect
{
	public:
	//Shape *shape;
	double dist, u, v;
	Ray ray;
	Point pos;
	Vector normal;
	ISect(){}
	ISect(Ray ray, double dist, Point pos, double u, double v, Vector normal) : ray(ray), dist(dist), pos(pos), u(u), v(v), normal(normal) {}
};

class Shape
{
	public:
	Image* texture; // index of the texture
	Color diffuse; // color
	Color emit;
	Color specular;
	virtual ISect* Intersect(Ray ray, double min, double max) = 0;
	virtual Vector Normal(Ray ray, Point pos) = 0;
	virtual Color Diffuse(Point pos, double u, double v)
	{
		if (texture == NULL)
	  		return diffuse;

		double nx = texture->Width;
		double ny = texture->Height;
		
		double g = u * (nx-1);
		double h = v * (ny-1);
		int x = int(g);
		int y = int(h);
		double s = g - x;
		double t = h - y;
		
		Color c00 = texture->Canvas[  y][  x];
		Color c01 = texture->Canvas[y+1][  x];
		Color c10 = texture->Canvas[  y][x+1];
		Color c11 = texture->Canvas[y+1][x+1];
		double A00 = (1-s)*(1-t);
		double A01 = (1-s)*(t);
		double A10 = (s)  *(1-t);
		double A11 = (s) * (t); 
		return c00 * A00 + c01 * A01 + c10 * A10 + c11 * A11; 
	}
	bool IsLight()
	{
		return emit.x > 1.e-6 || emit.y > 1.e-6 || emit.z > 1.e-6;
	}

	virtual Color Specular() 
	{ 
		if (IsLight()) 
			return Color(0,0,0); 
		else 
			return Color(.5,.5,.5); 
	}
	virtual Color Emit() { return emit; }
	virtual double SurfaceArea() = 0;
	virtual Point HitPoint(double u, double v) = 0;

};


class Sphere : public Shape
{
	public:
	double radius;  // radius
	Point center;  // center

  	Sphere(Point center, double radius, Color c, Color l, Image* texture = NULL) : center(center), radius(radius) {diffuse=c; emit = l; this->texture = texture; }

	static void FromFile(string fileName, vector<Shape*>& shapes, vector<Image*>& textures)
	{
		ifstream in(fileName.c_str());
		Point center;
		double radius;
		Color diffuse;
		Color emit;
		string option;
		string textureName;
		Image* texture = NULL;
		while (in >> center >> radius >> option)
		{
			if (option == "diffuse")
			{
				in >> diffuse;
				texture = NULL;
			}
			else if (option == "light")
				in >> emit; 
			else if (option == "texture")
			{	
				in >> textureName;
				texture = Image::SearchImages(textureName, textures);
			}
			else
				cout << "Unrecognized option: " << option << " Valid choices are 'diffuse', 'texture', 'light'." << endl;

			shapes.push_back(new Sphere(center, radius, diffuse, emit, texture));
		}
		in.close();
	}

	double SurfaceArea() 
	{ 
		return 4 * PI * pow(radius,2) * 1/2; // We only see 1/2 the sphere at any point.
	}  

	// compute the intersection of the ray with this sphere
	ISect* Intersect(Ray ray, double min, double max)
	{
		// solve (d.d)t^2 + 2d.(o-c)t + (o-c).(o-c) - R^2 = 0 for t
		Vector oc = ray.origin - center;
		double a = ray.dir.dot(ray.dir);
		double b = ray.dir.dot(oc);
		double c = oc.dot(oc) - pow(radius,2);
		double disc = pow(b,2) - a*c;
		if (disc < 0)
			return NULL;
		double t1 = (-b-sqrt(disc))/a;
		double t2 = (-b+sqrt(disc))/a;

		double minT;
		if (t1 >= min && t1 <= max)
			minT = t1;
		else if (t2 >= min && t2 <= max)
			minT = t2;
		else 
			return NULL;

		Point pos = ray.PointAtDistance(minT);

		Vector normal = Normal(ray, pos);
		/*
		double theta = acos( 0.999f * (pos.z-center.z)/radius ); 
		double sinTheta = sin(theta);
        double phi = acos( normal.x / (1.001f * sinTheta * radius) );
        if (pos.y - center.y < 0.0f) phi = 2*PI - phi;
        double u = phi/(2.0f*PI);
	    double v = 1.0f - theta/PI;
		*/

		double u = 1/(2*PI)*atan((pos.y-center.y)/(pos.x-center.x));
		double v = 1/PI * acos((pos.z-center.z)/radius);

		if (u<0)
			u = -u;

		if (u < 0) u = 0;
		if (u >= 1) u = .9999999999;
		if (v < 0) v = 0;
		if (v >= 1) v = .9999999999;

		return new ISect(ray, minT, pos, u, v, normal);

	}

	// Compute the smart normal that always faces towards origin of ray.
    Vector Normal(Ray ray, Point pos) 
	{
		Vector normal = (pos-center).norm();
		// If it's not facing origin of ray, reverse direction of normal
		if (normal.dot(-ray.dir) < 0)
			normal = -normal;
		return normal;
	}

	Point HitPoint(double u, double v) 
	{
		Point p;
		p.x = radius*sin(PI*v)*cos(2*PI*u) + center.x;
		p.y = radius*sin(PI*v)*sin(2*PI*u) + center.y;
		p.z = radius*cos(PI*v) + center.z;
		return p;
	}
};

class Triangle : public Shape
{
public:
	Point p0, p1, p2;
	double *u0, *u1, *u2;
	Triangle(Point p0, Point p1, Point p2, double* u0, double* u1, double* u2, Color diffuse, Color emit, Image* texture = NULL) : p0(p0), p1(p1), p2(p2), u0(u0), u1(u1), u2(u2) { this->diffuse = diffuse; this->emit = emit; this->texture = texture; }

	static void FromFile(string fileName, vector<Shape*>& shapes, vector<Image*>& textures)
	{
		ifstream in(fileName.c_str());
		Point p0, p1, p2;
		double *u0, *u1, *u2;
		string textureName, option;
		Color diffuse, emit;
		Image* texture = NULL;

		u0 = new double[2]; u1 = new double[2]; u2 = new double[2];
		while (in >> p0 >> u0[0] >> u0[1] >> p1 >> u1[0] >> u1[1] >> p2 >> u2[0] >> u2[1] >> option)
		{
			if (option == "diffuse")
				in >> diffuse;
			else if (option == "light")
				in >> emit; 
			else if (option == "texture")
			{	
				in >> textureName;
				texture = Image::SearchImages(textureName, textures);
			}
			else
				cout << "Unrecognized option: " << option << " Valid choices are 'diffuse', 'texture', 'light'." << endl;

			shapes.push_back(new Triangle(p0, p1, p2, u0, u1, u2, diffuse, emit, texture));
			u0 = new double[2]; u1 = new double[2]; u2 = new double[2];
		}
		in.close();
	}

	double SurfaceArea()
	{
		return (p1-p0).cross(p2-p0).length()/2.0;
	}

    ISect* Intersect(Ray ray, double min, double max)
	{ 
		Vector w1 = p0-p1;
		Vector w2 = p0-p2;
		Vector w3 = ray.dir;
		Vector w4 = p0-ray.origin;

		double A = w1.x, D = w2.x, G = w3.x, J = w4.x;
		double B = w1.y, E = w2.y, H = w3.y, K = w4.y;
		double C = w1.z, F = w2.z, I = w3.z, L = w4.z;
		
		double EIHF = E*I-H*F;
		double GFDI = G*F-D*I;
		double DHEG = D*H-E*G;
		double denom = (A*EIHF + B*GFDI + C*DHEG);
		double beta = (J*EIHF + K*GFDI + L*DHEG) / denom;
		if (beta < 0.0 || beta > 1.0) 
			return NULL;
	
		double AKJB = A*K - J*B;
		double JCAL = J*C - A*L;
		double BLKC = B*L - K*C;
		double gamma = (I*AKJB + H*JCAL + G*BLKC)/denom;
		if (gamma < 0.0 || beta + gamma > 1.0) 
			return NULL;
		
		double t =  -(F*AKJB + E*JCAL + D*BLKC)/denom;
		if (t < min || t > max)
			return NULL;
		
		double alpha = 1 - beta - gamma;
		Point pos = p0 * alpha + p1 * beta + p2 * gamma;	
		double u = alpha * u0[0] + beta * u1[0] + gamma * u2[0];
		double v = alpha * u0[1] + beta * u1[1] + gamma * u2[1];
		if (u < 0) u = 0;
		if (u >= 1) u = .9999999999;
		if (v < 0) v = 0;
		if (v >= 1) v = .9999999999;

		return new ISect(ray, t, pos, u, v, Normal(ray, pos));
	}

	// Smart Normal
	Vector Normal(Ray ray, Point pos) 
	{
		Vector normal = (p1-p0).cross(p2-p0).norm();
		// If it's not facing origin of ray, reverse direction of normal
		if (normal.dot(-ray.dir) < 0)
			normal = -normal;
		return normal;
	}

	Point HitPoint(double u, double v) 
	{
		if (u+v > .9999999999)
		{
			u = 1-u;
			v = 1-v;
		}
		return(p0 + (p1-p0)*u + (p2-p0)*v);
	}

};

class Camera
{
public:
	Vector u,v,w;
	Point eye;
	double d, vfov;
	Camera(){}
	Camera(Point lookfrom, Point lookat, Vector vup, double _vfov, double _d)
	{
		w = -(lookat-lookfrom).norm();
		u = vup.cross(w).norm();
		v = w.cross(u).norm();
		d = _d;
		vfov = _vfov;
		eye = lookfrom;
	}
};


class Matrix
{
public:
	double elem[4][4];

	Matrix operator*(const Matrix& b)
	{
	  Matrix m;
	  register double sum;

	  for (int j=0; j<4;  j++) 
	    for (int i=0; i<4; i++) {
	      sum = 0.0;
	      for (int k=0; k<4; k++)
		sum +=  elem[j][k] * b.elem[k][i];
	      m.elem[j][i] = sum;
	    }
	  return m;
	}

	// returns a 4x4 identity matrix
	static Matrix Identity()
	{
	  Matrix m;
	  for (int i=0; i<4; i++)   // set all elements to zero
	    for (int j=0; j<4; j++)
	      m.elem[i][j] = 0.0;

	  for (int i=0; i<4; i++)   // set diagonal elements to one
	    m.elem[i][i] = 1.0;

	  return m;
	}

	// returns a 4x4 scale matrix, given sx,sy,sz as inputs 
	static Matrix Scale(double sx, double sy, double sz)
	{
	  Matrix m;
	  m = Identity();
	  m.elem[0][0] = sx;
	  m.elem[1][1] = sy;
	  m.elem[2][2] = sz;
	  return m;
	}
	    
	// returns a 4x4 translation matrix, given tx,ty,tz as inputs 
	static Matrix Translate(double tx, double ty, double tz)
	{
	  Matrix m;
	  m = Identity();
	  m.elem[0][3] = tx; m.elem[1][3] = ty; m.elem[2][3] = tz;
	  return m;
	}

	static Matrix Rotate(Vector u, Vector v, Vector w)
	{
		Matrix m = 
		   {{{u.x, v.x, w.x, 0.0}, 
		     {u.y, v.y, w.y, 0.0},
		     {u.z, v.z, w.z, 0.0}, 
			 {0.0, 0.0, 0.0, 1.0}}};
		return m;
	}
};

// transform a vector by matrix
Vector operator* (const Matrix& left_op, const Vector& right_op)
{
   Vector ret;
   double temp;
   ret.x = right_op.x * left_op.elem[0][0] + right_op.y * left_op.elem[0][1] + 
              right_op.z * left_op.elem[0][2] +               left_op.elem[0][3];
   ret.y = right_op.x * left_op.elem[1][0] + right_op.y * left_op.elem[1][1] +
              right_op.z * left_op.elem[1][2] +               left_op.elem[1][3];
   ret.z = right_op.x * left_op.elem[2][0] + right_op.y * left_op.elem[2][1] +
              right_op.z * left_op.elem[2][2] +               left_op.elem[2][3];
   temp   = right_op.x * left_op.elem[3][0] + right_op.y * left_op.elem[3][1] +
              right_op.z * left_op.elem[3][2] +               left_op.elem[3][3];
   ret = ret * (1.0/temp); // DeHomogenize
   return ret;
}


class RayTracer
{
public:
	Camera camera;
	vector<Shape*> shapes;
	vector<Shape*> lights;
	vector<Image*> textures;
	Image image;
	Color background;
	Color ambient;
	int sqrtNSamples;
	Matrix transform;
	int lightSamples;
	int depth;

	RayTracer(int nx, int ny, string inputFile)
	{
		Point lookfrom, lookat;
		double vfov;
		Vector vup;
		string shapeFileName;

		ifstream in(inputFile.c_str());
		string setting;
		while (in >> setting)
		{
			if (setting == "lookfrom") in >> lookfrom;
			else if (setting == "lookat") in >> lookat;
			else if (setting == "vfov") in >> vfov;
			else if (setting == "vup") in >> vup;
			else if (setting == "background") in >> background;
			else if (setting == "ambient") in >> ambient;
			else if (setting == "pixelsamples") in >> sqrtNSamples;
			else if (setting == "lightsamples") in >> lightSamples;
			else if (setting == "depth") in >> depth;
			else if (setting == "spheres")
			{
				in >> shapeFileName;
				Sphere::FromFile(shapeFileName, shapes, textures);
			}
			else if (setting == "triangles")
			{
				in >> shapeFileName;
				Triangle::FromFile(shapeFileName, shapes, textures);
			}
			else
				cout << "INVALID COMMAND! " << setting;
		}
		in.close();

		for (unsigned int i = 0; i<shapes.size(); ++i)
		{
			if (shapes[i]->IsLight())
				lights.push_back(shapes[i]);
		}

		camera = Camera(lookfrom, lookat, vup, vfov * PI/180.0, 1); // Convert vfov from degrees to radians.
		image = Image(nx, ny);

		double h = 2 * camera.d * tan(camera.vfov/2.0);
		Matrix M1 = Matrix::Translate(-image.Width/2.0, -image.Height/2.0, -camera.d);
		Matrix M2 = Matrix::Scale((double) h/image.Height, (double) h/image.Height, 1);
		Matrix M3 = Matrix::Rotate(camera.u, camera.v, camera.w);
		Matrix M4 = Matrix::Translate(camera.eye.x, camera.eye.y, camera.eye.z);

		transform = M4 * M3 * M2 * M1;
	}


	void Render()
	{
		double pct = 0, lastpct = 0;
		for (int y=0; y<image.Height; ++y)
			for (int x=0; x<image.Width; ++x)
			{
				pct = (double) (y* image.Width + x)/(image.Height * image.Width) * 100;
				if (int(pct) > int(lastpct))
					cout << int(pct) << "%\t";
				lastpct = pct;

				Color pixel = Color(0,0,0);

				// Box Filtering
				for (int k=0; k<sqrtNSamples; ++k)
					for (int l=0; l<sqrtNSamples; ++l)	
					{			
						double u = (l+randf())/sqrtNSamples;
						double v = (k+randf())/sqrtNSamples;
						Point p(x+u, y+v, 0);
						Point pp = transform * p;  // Transform p to new coordinate system.  
						Ray ray(camera.eye, pp - camera.eye);
					  	pixel = pixel + RayColor(ray, depth);
					}
				
				pixel = pixel / (sqrtNSamples * sqrtNSamples);
				image.Canvas[y][x] = pixel;
			}
	}

	bool IsDark(Color c)
	{
		return c.x < 1e-10 && c.y < 1e-10 && c.z < 1e-10;
	}

	void MinIntersect(ISect*& iSect, Shape*& shape, const vector<Shape*>& shapes, const Ray& ray, double min, double max)
	{
		iSect = NULL;
		shape = NULL;
		for (unsigned int i=0; i<shapes.size(); ++i)
		{
			ISect* sect = shapes[i]->Intersect(ray, min, max);
			if (sect != NULL && (iSect == NULL || sect->dist < iSect->dist))
			{
				iSect = sect;
				shape = shapes[i];
			}
		}
	}


	Color RayColor(const Ray& ray, int depth)
	{
		if (depth <= 0)
			return Color(0,0,0);

		ISect* iSect = NULL;
		Shape* shape = NULL;
		MinIntersect(iSect, shape, shapes, ray, camera.d, INFINITY);

		if (iSect == NULL)
			return background;
			
		Color diffuse = shape->Diffuse(iSect->pos, iSect->u, iSect->v);
		Color specular = shape->Specular();
		Color emit = shape->Emit();
		
		Color color = emit + diffuse * ambient;
		if (!IsDark(diffuse))
		{
			//if (diffuse == Vector(0,1,0))
			//{	
				color += diffuse * DirectLight(iSect);
			//	return color;
			//}
		}
		if (!IsDark(specular))
		{
			Ray reflect(iSect->pos, ray.dir.reflect(iSect->normal));
			color += specular * RayColor(reflect, depth-1);
		}
		
		// Lambertian Lighting
		//Vector l = (light-iSect->pos).norm();
		//Vector n = iSect->normal;
		//Color color = diffuse * ambient + diffuse * max<double>(0, l.dot(n)); 

		return color;
	}

	Color DirectLight(ISect* source)
	{
		Color totalLight, light;
		ISect* target = NULL;
		Shape* shape;
		double u, v;
		double max;
		Point pos;
		Vector toLight;

		for (unsigned int i=0; i<lights.size(); ++i)
		{
			light = Color(0,0,0);
			// Take multiple samples of the light.
			for (int m=0; m < lightSamples; ++m)
				for (int n=0; n < lightSamples; ++n)
				{
					u = (double) (n + randf())/lightSamples;
					v = (double) (m + randf())/lightSamples;
					Point pos = lights[i]->HitPoint(u,v);
					Ray lightRay(source->pos, pos - source->pos);
					
					target = lights[i]->Intersect(lightRay, 0, INFINITY);
					// target "should* intersect light!! (not sure why it doesn't some times).
					if (target == NULL)
						continue;
					pos = target->pos; // If the light is a sphere, make sure the point of intersection is the closest position!

					// Test to see if ray intersects any intermediate objects before light.
					MinIntersect(target, shape, shapes, lightRay, 0.0001, INFINITY); // Make sure it doesn't intersect itself!!! (use min .00001 , not min = 0).

					Point pos2 = target->pos;
					if (abs(pos.x-pos2.x)< .00001 && abs(pos.y-pos2.y)<.00001 && abs(pos.z-pos2.z) < .000001) // No intermediate objects!
					{
						toLight = source->pos - target->pos;
						light += 1/PI * lightRay.dir.dot(source->normal) * /*(-lightRay.dir).dot(target->normal) **/ shape->SurfaceArea() * shape->Emit() * 1/pow(toLight.length(),2);
					}
				}
			light /= (lightSamples*lightSamples); // average light.
			totalLight += light;
		}
		if (totalLight.x > 1) totalLight.x = 1;
		if (totalLight.y > 1) totalLight.y = 1;
		if (totalLight.z > 1) totalLight.z = 1;
		return totalLight;
	}
};


int main(int argc, char** argv)
{
	int nx = atoi(argv[1]), ny = atoi(argv[2]);
	char *inputFile = argv[3], *outputFile = argv[4];
	//int nx = 512, ny = 512;	
	//char *inputFile = "settings.txt", *outputFile = "output.ppm";
	RayTracer tracer(nx, ny, inputFile);
	tracer.Render();
	tracer.image.Output(outputFile);
	return 0;
}




