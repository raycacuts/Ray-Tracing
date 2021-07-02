#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>

#include <fmt/printf.h>
#include <stb_image.h>
#include <stb_image_write.h>

#include <limits>
#include <memory>
#include <vector>
#include <atlas/utils/LoadObjFile.hpp>
#include <atlas/math/Solvers.hpp>

using atlas::core::areEqual;

using Colour = atlas::math::Vector;
using Ray = atlas::math::Ray<atlas::math::Vector>;

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

// Declarations
class BRDF;
class Camera;
class Material;
class Light;
class Shape;
class Sampler;
class World;
class Tracer;

struct ShadeRec
{
	bool hit_object;
	Colour color;
	int depth;
	float tmin;
	atlas::math::Normal normal;
	atlas::math::Ray<atlas::math::Vector> ray;
	atlas::math::Point hit_point;
	atlas::math::Point local_hit_point;
	std::shared_ptr<Material> material;
	std::shared_ptr<World> world;
};


class World
{
public:
	std::size_t width, height;
	Colour background;
	std::shared_ptr<Sampler> sampler;
	std::vector<std::shared_ptr<Shape>> objects;
	std::vector<Colour> image;
	std::vector<std::shared_ptr<Light>> lights;
	std::shared_ptr<Light> ambient;

	std::shared_ptr<Camera> camera;
	std::shared_ptr<Tracer> tracer;

	int max_depth;

public:
	World();
	World(std::size_t pWidth, std::size_t pHeight, Colour pBackground, int depth);
	void setSampler(std::shared_ptr<Sampler> const& pSampler);
	void setCamera(std::shared_ptr<Camera> const& pCamera);
	void setTracer(std::shared_ptr<Tracer> const& pTracer);

	void addObject(std::shared_ptr<Shape> const& pObject);
	void build();
	ShadeRec hit_objects(const Ray& ray);

};

class Tracer {
public:
	Tracer();
	Tracer(std::shared_ptr<World> pWorld);
	virtual Colour trace_ray(const Ray& ray) const;
	virtual Colour trace_ray(const Ray ray, const int depth) const;
	virtual Colour trace_ray(const Ray ray, float& tmin, const int depth) const;

protected:
	std::shared_ptr<World> world;
};
class PathTrace : public Tracer
{
public:
	PathTrace();
	PathTrace(std::shared_ptr<World> pWorld);
	Colour trace_ray(const Ray& ray) const override;
	Colour trace_ray(const Ray ray, const int depth) const override;
	Colour trace_ray(const Ray ray, float& tmin, const int depth) const override;
};
class Sampler
{
public:
	Sampler(int numSamples, int numSets);
	virtual ~Sampler() = default;

	int getNumSamples() const;

	void setupShuffledIndeces();

	void mapSamplesToHemisphere(const float e);
	virtual void generateSamples() = 0;

	atlas::math::Point sampleUnitSquare();
	atlas::math::Point sampleHemiSphere();

protected:
	std::vector<atlas::math::Point> mSamples;
	std::vector<atlas::math::Point> mHemisphereSamples;
	std::vector<int> mShuffledIndeces;

	int mNumSamples;
	int mNumSets;
	unsigned long mCount;
	int mJump;
};
class Regular : public Sampler
{
public:
	Regular(int numSamples, int numSets);

	void generateSamples();
};

class Random : public Sampler
{
public:
	Random(int numSamples, int numSets);

	void generateSamples();
};
class Jittered : public Sampler
{
public:
	Jittered(int numSamples, int numSets);
	void generateSamples();
};
// Abstract classes defining the interfaces for concrete entities
class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	virtual void renderScene() const = 0;

	void setEye(atlas::math::Point const& eye);

	void setLookAt(atlas::math::Point const& lookAt);

	void setUpVector(atlas::math::Vector const& up);

	void computeUVW();

	void setWorld(std::shared_ptr<World>& world);

protected:
	atlas::math::Point mEye;
	atlas::math::Point mLookAt;
	atlas::math::Point mUp;
	atlas::math::Vector mU, mV, mW;
	std::shared_ptr<World> world;
};
class Pinhole : public Camera
{
public:
	Pinhole();

	void setDistance(float distance);
	void setZoom(float zoom);

	atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
	void renderScene() const override;

private:
	float mDistance;
	float mZoom;
};


class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be
	// updated in sr
	virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) = 0;

	void setColour(Colour const& col);

	Colour getColour() const;

	std::shared_ptr<Material> getMaterial();
	void setMaterial(std::shared_ptr<Material> const& material);

	std::shared_ptr<Material> getMaterial() const;

	virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const = 0;

	void setSampler(std::shared_ptr<Sampler> sampler);
	virtual atlas::math::Point sample();

	virtual float pdf(ShadeRec& sr) = 0;
	virtual atlas::math::Vector getNormal(atlas::math::Point const& p) = 0;
protected:

	atlas::math::Vector normal;
	std::shared_ptr<Sampler> sampler_ptr;
	Colour mColour;
	std::shared_ptr<Material> mMaterial;
};
class Triangle : public Shape
{
public:
	Triangle();
	Triangle(atlas::math::Point const& a, atlas::math::Point const& b, atlas::math::Point const& c);
	Triangle(atlas::utils::Vertex const& x, atlas::utils::Vertex const& y,
		atlas::utils::Vertex const& z);

	void setA(atlas::math::Point const& a);
	void setB(atlas::math::Point const& b);
	void setC(atlas::math::Point const& c);
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) ;
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point sample() override;
	float pdf(ShadeRec& sr) override;
	atlas::math::Vector getNormal(atlas::math::Point const& p) override;
private:
	atlas::math::Point mA;
	atlas::math::Point mB;
	atlas::math::Point mC;
};
class Rectangle : public Shape
{
public:
	Rectangle();
	Rectangle(atlas::math::Point const& po, atlas::math::Vector const& a, atlas::math::Vector const& b);

	void setPo(atlas::math::Point const& po);
	void setA(atlas::math::Vector const& a);
	void setB(atlas::math::Vector const& b);
	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) ;
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point sample() override;
	float pdf(ShadeRec& sr) override;
	atlas::math::Vector getNormal(atlas::math::Point const& p) override;

private:
	atlas::math::Point mPo;
	atlas::math::Vector mA;
	atlas::math::Vector mB;
};
class Sphere : public Shape
{
public:
	Sphere(atlas::math::Point center, float radius);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) ;
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point sample() override;
	float pdf(ShadeRec& sr) override;
	atlas::math::Vector getNormal(atlas::math::Point const& p) override;
private:

	atlas::math::Point mCentre;
	float mRadius;
	float mRadiusSqr;
};
class Plane : public Shape
{
public:
	Plane(atlas::math::Vector a, atlas::math::Vector n);

	bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
		ShadeRec& sr) ;
	bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
		float& tMin) const;

	atlas::math::Point sample() override;
	float pdf(ShadeRec& sr) override;
	atlas::math::Vector getNormal(atlas::math::Point const& p) override;
private:

	atlas::math::Vector mA;
	atlas::math::Vector mN;

};
class BTDF {
public:
	virtual ~BTDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;

	virtual Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& wt) const = 0;

	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;

	virtual bool tir(const ShadeRec& sr) const = 0;

	virtual void setSampler(std::shared_ptr<Sampler> sampler) = 0;
};
class PerfectTransmitter :BTDF {
public:
	PerfectTransmitter();
	PerfectTransmitter(float ior, float kt);
	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& wt) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	bool tir(const ShadeRec& sr) const override;

	void setSampler(std::shared_ptr<Sampler> sampler) override;
	void setIor(float ior);
	void setKt(float kt);
	void setColour(Colour color);
private:
	Colour mColour;
	float mIor;
	float mKt;
	std::shared_ptr<Sampler> mSampler;
};
class FresnelTransmitter :BTDF {
public:
	FresnelTransmitter();
	FresnelTransmitter(float etain, float etaout);
	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& wt) const override;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	float fresnel(const ShadeRec& sr) const;
	bool tir(const ShadeRec& sr) const override;

	void setSampler(std::shared_ptr<Sampler> sampler) override;
	void setEtaIn(float etain);
	void setEtaOut(float etaout);
	void setColour(Colour color);
private:
	Colour mColour;
	float mEtaIn;
	float mEtaOut;
	std::shared_ptr<Sampler> mSampler;
};
class BRDF
{
public:
	virtual ~BRDF() = default;

	virtual Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const = 0;

	virtual Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const = 0;

	virtual Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const = 0;

	virtual void setSampler(std::shared_ptr<Sampler> sampler) = 0;
};
class FresnelReflector :BRDF {
public:
	FresnelReflector();
	FresnelReflector(float etain, float etaout);
	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& wt) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& wt, float& pdf) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	float fresnel(const ShadeRec& sr) const;

	void setSampler(std::shared_ptr<Sampler> sampler) override;
	void setEtaIn(float etain);
	void setEtaOut(float etaout);
	void setColour(Colour color);
private:
	Colour mColour;
	float mEtaIn;
	float mEtaOut;
	std::shared_ptr<Sampler> mSampler;
};
class GlossySpecular : public BRDF
{
public:
	GlossySpecular();
	GlossySpecular(Colour specularColour, float diffuseReflection, float specularReflection, float exponent);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming, float& pdf) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setSampler(std::shared_ptr<Sampler> sampler) override;

	void setDiffuseReflection(float kd);
	void setSpecularReflection(float ks);
	void setSpecularColour(Colour const& colour);
	void setKr(float pkr);
	void setExponent(float ep);
private:
	Colour mSpecularColour;
	float mDiffuseReflection;
	float mSpecularReflection;
	float mExponent;
	float kr;
	std::shared_ptr<Sampler> mSampler;
};

class PerfectSpecular : public BRDF
{
public:
	PerfectSpecular();
	PerfectSpecular(Colour pPerfectColour, float pkr);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming, float& pdf) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setSampler(std::shared_ptr<Sampler> sampler) override;
	void setKr(float kr);
	void setPerfectColour(Colour const& colour);
private:
	Colour mPerfectColour;
	float kr;
	std::shared_ptr<Sampler> mSampler;
};
class Lambertian : public BRDF
{
public:
	Lambertian();
	Lambertian(Colour diffuseColor, float diffuseReflection);

	Colour fn(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector const& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming) const override;

	Colour sample_f(ShadeRec const& sr,
		atlas::math::Vector const& reflected,
		atlas::math::Vector& incoming, float& pdf) const;

	Colour rho(ShadeRec const& sr,
		atlas::math::Vector const& reflected) const override;

	void setDiffuseReflection(float kd);

	void setDiffuseColour(Colour const& colour);

	void setSampler(std::shared_ptr<Sampler> sampler) override;
private:
	Colour mDiffuseColour;
	float mDiffuseReflection;
	std::shared_ptr<Sampler> mSampler;

};
class Material
{
public:
	virtual ~Material() = default;

	virtual Colour L(ShadeRec& sr) = 0;
	virtual Colour shade(ShadeRec& sr) = 0;
	virtual Colour areaLightShade(ShadeRec& sr) = 0;
	virtual Colour pathShade(ShadeRec& sr) = 0;

private:
};


// Concrete classes which we can construct and use in our ray tracer



class Emissive : public Material
{
public:
	Emissive();
	Emissive(float radiance, Colour color);

	void scaleRadiance(float b);
	void setColour(Colour const& c);

	Colour L(ShadeRec& sr)  override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;

private:
	Colour mColour;
	float mRadiance;
};
class Matte : public Material
{
public:
	Matte();
	Matte(float kd, float ka, Colour color);

	void setDiffuseReflection(float k);

	void setAmbientReflection(float k);

	void setDiffuseColour(Colour colour);

	Colour L(ShadeRec& sr)  override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;
	void setSampler(std::shared_ptr<Sampler> sampler);

private:

	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<Lambertian> mAmbientBRDF;
};
class Phong : public Material
{
public:
	Phong();
	Phong(float ks, float kd, float ka, Colour color, float exponent);

	void setDiffuseReflection(float k);
	void setAmbientReflection(float k);
	void setSpecularReflection(float k);
	void setExponent(float k);
	void setKr(float kr);

	void setSpecularColour(Colour colour);
	Colour L(ShadeRec& sr)  override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;
	float getExponent();
	void setSampler(std::shared_ptr<Sampler> sampler);

private:
	float mExponent;
	std::shared_ptr<Lambertian> mAmbientBRDF;
	std::shared_ptr<Lambertian> mDiffuseBRDF;
	std::shared_ptr<GlossySpecular> mSpecularBRDF;
};
class Dielectric : public Phong
{
public:
	Dielectric();
	Dielectric(Colour mColourIn, Colour mColourOut);

	void setEtaIn(float etain);
	void setEtaOut(float etaout);
	void setColourIn(Colour colorin);
	void setColourOut(Colour colorout);
	void setSampler(std::shared_ptr<Sampler> sampler);

	Colour L(ShadeRec& sr)  override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;

private:
	Colour mColourIn;
	Colour mColourOut;

	std::shared_ptr<FresnelReflector> mFresnelBRDF;
	std::shared_ptr<FresnelTransmitter> mFresnelBTDF;
};
class Transparent : public Phong
{
public:
	Transparent();
	Transparent(float ior, float kr, float kt);
	Colour L(ShadeRec& sr)  override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;
	void setSampler(std::shared_ptr<Sampler> sampler);

	void setIor(float ior);
	void setKr(float kr);
	void setKt(float kt);

	void setColour(Colour color);
private:
	std::shared_ptr<PerfectSpecular> mReflectiveBRDF;
	std::shared_ptr<PerfectTransmitter> mSpecularBTDF;

};
class Reflective : public Phong
{
public:
	Reflective();
	Reflective(float kr, Colour color);

	Colour L(ShadeRec& sr) override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;
	void setSampler(std::shared_ptr<Sampler> sampler);

	void setKr(float kr);
private:
	std::shared_ptr<PerfectSpecular> mPerfectBRDF;
};
class GlossyReflector : public Phong
{
public:
	GlossyReflector();
	GlossyReflector(float kr, Colour color);

	Colour L(ShadeRec& sr) override;
	Colour shade(ShadeRec& sr) override;
	Colour pathShade(ShadeRec& sr) override;
	Colour areaLightShade(ShadeRec& sr) override;
	void setSampler(std::shared_ptr<Sampler> sampler);

	void setKs(float ks);
	void setKa(float ka);
	void setKd(float kd);
	void setColour(Colour color);
	void setKr(float kr);
	void setExponent(float ex);

private:
	std::shared_ptr<GlossySpecular> mGlossyBRDF;
};
class Light
{
public:
	Light();
	virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

	virtual Colour L(ShadeRec& sr) = 0;

	virtual bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

	virtual float G(ShadeRec& sr) = 0;
	virtual float pdf(ShadeRec& sr) = 0;

	void scaleRadiance(float b);

	void setColour(Colour const& c);


protected:
	Colour mColour;
	float mRadiance;
};
class AmbientOccluder : public Light
{
public:
	AmbientOccluder();

	void setMinAmount(float pMinAmount);
	void setSampler(std::shared_ptr<Sampler> sampler);
	atlas::math::Vector getDirection(ShadeRec& sr) override;
	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;

	Colour L(ShadeRec& sr)  override;
	float G(ShadeRec& sr) override;
	float pdf(ShadeRec& sr) override;

private:
	atlas::math::Vector mU, mV, mW;
	std::shared_ptr<Sampler> mSampler;
	float minAmount;
};
class AreaLight : public Light
{
public:
	AreaLight();
	AreaLight(std::shared_ptr<Shape> const& object, std::shared_ptr<Material> const& material);
	void setObject(std::shared_ptr<Shape> const& object);
	void setMaterial(std::shared_ptr<Material> const& material);

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;

	Colour L(ShadeRec& sr)  override;
	float G(ShadeRec& sr) override;
	float pdf(ShadeRec& sr) override;
	atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
	std::shared_ptr<Shape> mObject;
	std::shared_ptr<Material> mMaterial;
	atlas::math::Point samplePoint;
	atlas::math::Vector normal;
	atlas::math::Vector wi;

};
class PointLight : public Light
{
public:
	PointLight();
	PointLight(atlas::math::Point const& location);

	Colour L(ShadeRec& sr)  override;
	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;
	void setLocation(atlas::math::Point const& location);
	atlas::math::Vector getDirection(ShadeRec& sr) override;

	float G(ShadeRec& sr)override;
	float pdf(ShadeRec& sr) override;
private:
	atlas::math::Point mLocation;
};
class Directional : public Light
{
public:
	Directional();
	Directional(atlas::math::Vector const& d);

	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;
	void setDirection(atlas::math::Vector const& d);
	atlas::math::Vector getDirection(ShadeRec& sr) override;

	Colour L(ShadeRec& sr)  override;
	float G(ShadeRec& sr)override;
	float pdf(ShadeRec& sr) override;
private:
	atlas::math::Vector mDirection;
};

class Ambient : public Light
{
public:
	Ambient();

	atlas::math::Vector getDirection(ShadeRec& sr) override;
	bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;

	Colour L(ShadeRec& sr)  override;
	float G(ShadeRec& sr)override;
	float pdf(ShadeRec& sr) override;
private:
	atlas::math::Vector mDirection;
};
