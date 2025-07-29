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

using atlas::core::areEqual;

using Colour = atlas::math::Vector;

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

struct World
{
    std::size_t width, height;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
    std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
};

struct ShadeRec
{
    Colour color;
    float t;
    atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    atlas::math::Point hit_point;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
};
class Sampler
{
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndeces();

    virtual void generateSamples() = 0;

    atlas::math::Point sampleUnitSquare();

protected:
    std::vector<atlas::math::Point> mSamples;
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

    virtual void renderScene(std::shared_ptr<World>& world) const = 0;

    void setEye(atlas::math::Point const& eye);

    void setLookAt(atlas::math::Point const& lookAt);

    void setUpVector(atlas::math::Vector const& up);

    void computeUVW();

protected:
    atlas::math::Point mEye;
    atlas::math::Point mLookAt;
    atlas::math::Point mUp;
    atlas::math::Vector mU, mV, mW;
};
class Pinhole : public Camera
{
public:
    Pinhole();

    void setDistance(float distance);
    void setZoom(float zoom);

    atlas::math::Vector rayDirection(atlas::math::Point const& p) const;
    void renderScene(std::shared_ptr<World>& world) const override;

private:
    float mDistance;
    float mZoom;
};
class Fisheye : public Camera
{
public:
    Fisheye();

    void setPsi_max(float psi_max);
    atlas::math::Vector rayDirection(atlas::math::Point const& p, ShadeRec& sr, float& r_squared) const;
    void renderScene(std::shared_ptr<World>& world) const override;

private:
    float mPsi_max;
};


class Shape
{
public:
    Shape();
    virtual ~Shape() = default;

    // if t computed is less than the t in sr, it and the color should be
    // updated in sr
    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const = 0;

    void setColour(Colour const& col);

    Colour getColour() const;

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

    void setA(atlas::math::Point const& a);
    void setB(atlas::math::Point const& b);
    void setC(atlas::math::Point const& c);
    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;
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
        ShadeRec& sr) const;
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
        ShadeRec& sr) const;
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
        ShadeRec& sr) const;
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    atlas::math::Point sample() override;
    float pdf(ShadeRec& sr) override;
    atlas::math::Vector getNormal(atlas::math::Point const& p) override;
private:

    atlas::math::Vector mA;
    atlas::math::Vector mN;

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

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

    void setDiffuseReflection(float kd);
    void setSpecularReflection(float ks);
    void setSpecularColour(Colour const& colour);
private:
    Colour mSpecularColour;
    float mDiffuseReflection;
    float mSpecularReflection;
    float mExponent;
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

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

    void setDiffuseReflection(float kd);

    void setDiffuseColour(Colour const& colour);

private:
    Colour mDiffuseColour;
    float mDiffuseReflection;
};
class Material
{
public:
    virtual ~Material() = default;

    virtual Colour L(ShadeRec& sr) = 0;
    virtual Colour shade(ShadeRec& sr) = 0;
    virtual Colour areaLightShade(ShadeRec& sr) = 0;
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
    Colour areaLightShade(ShadeRec& sr) override;

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


    void setSpecularColour(Colour colour);
    Colour L(ShadeRec& sr)  override;
    Colour shade(ShadeRec& sr) override;
    Colour areaLightShade(ShadeRec& sr) override;

private:
    float mExponent;
    std::shared_ptr<Lambertian> mAmbientBRDF;
    std::shared_ptr<Lambertian> mDiffuseBRDF;
    std::shared_ptr<GlossySpecular> mSpecularBRDF;
};

class Light
{
public:
    Light();
    virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

    virtual Colour L(ShadeRec& sr) const = 0;

    virtual bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const = 0;

    virtual float G(ShadeRec& sr) = 0;
    virtual float pdf(ShadeRec& sr) = 0;

    void scaleRadiance(float b);

    void setColour(Colour const& c);


protected:
    Colour mColour;
    float mRadiance;
};
class AreaLight : public Light
{
public:
    AreaLight();
    AreaLight(std::shared_ptr<Shape> const& object, std::shared_ptr<Material> const& material);
    void setObject(std::shared_ptr<Shape> const& object);
    void setMaterial(std::shared_ptr<Material> const& material);

    bool in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const override;

    Colour L(ShadeRec& sr) const override;
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

    Colour L(ShadeRec& sr) const override;
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

    Colour L(ShadeRec& sr) const override;
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

    Colour L(ShadeRec& sr) const override;
    float G(ShadeRec& sr)override;
    float pdf(ShadeRec& sr) override;
private:
    atlas::math::Vector mDirection;
};

