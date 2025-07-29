#include "assignment.hpp"

const float kEpsilon{ 0.05f };
// ******* Function Member Implementation *******

// ***** Sampler function members *****
Sampler::Sampler(int numSamples, int numSets) :
    mNumSamples{ numSamples }, mNumSets{ numSets }, mCount{ 0 }, mJump{ 0 }
{
    mSamples.reserve(mNumSets* mNumSamples);
    setupShuffledIndeces();
}

int Sampler::getNumSamples() const
{
    return mNumSamples;
}

void Sampler::setupShuffledIndeces()
{
    mShuffledIndeces.reserve(mNumSamples * mNumSets);
    std::vector<int> indices;

    std::random_device d;
    std::mt19937 generator(d());

    for (int j = 0; j < mNumSamples; ++j)
    {
        indices.push_back(j);
    }

    for (int p = 0; p < mNumSets; ++p)
    {
        std::shuffle(indices.begin(), indices.end(), generator);

        for (int j = 0; j < mNumSamples; ++j)
        {
            mShuffledIndeces.push_back(indices[j]);
        }
    }
}

atlas::math::Point Sampler::sampleUnitSquare()
{
    if (mCount % mNumSamples == 0)
    {
        atlas::math::Random<int> engine;
        mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
    }

    return mSamples[mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]];
}
// ***** Jittered function memebers *****
Jittered::Jittered(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}
void Jittered::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));
    atlas::math::Random<float> engine;
    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ (float)q / (float)n + engine.getRandomOne() / (float)n,
                    (float)p / (float)n + engine.getRandomOne() / (float)n, 0.0f });
            }
        }
    }
}
// ***** Regular function members *****
Regular::Regular(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Regular::generateSamples()
{
    int n = static_cast<int>(glm::sqrt(static_cast<float>(mNumSamples)));

    for (int j = 0; j < mNumSets; ++j)
    {
        for (int p = 0; p < n; ++p)
        {
            for (int q = 0; q < n; ++q)
            {
                mSamples.push_back(
                    atlas::math::Point{ (q + 0.5f) / n, (p + 0.5f) / n, 0.0f });
            }
        }
    }
}

// ***** Regular function members *****
Random::Random(int numSamples, int numSets) : Sampler{ numSamples, numSets }
{
    generateSamples();
}

void Random::generateSamples()
{
    atlas::math::Random<float> engine;
    for (int p = 0; p < mNumSets; ++p)
    {
        for (int q = 0; q < mNumSamples; ++q)
        {
            mSamples.push_back(atlas::math::Point{
                engine.getRandomOne(), engine.getRandomOne(), 0.0f });
        }
    }
}

// ***** Camera function members *****
Camera::Camera() :
    mEye{ 0.0f, 0.0f, 500.0f },
    mLookAt{ 0.0f, 0.0f, 0.0f },
    mUp{ 0.0f, 1.0f, 0.0f },
    mU{ 1.0f, 0.0f, 0.0f },
    mV{ 0.0f, 1.0f, 0.0f },
    mW{ 0.0f, 0.0f, 1.0f }
{}
void Camera::setEye(atlas::math::Point const& eye)
{
    mEye = eye;
}

void Camera::setLookAt(atlas::math::Point const& lookAt)
{
    mLookAt = lookAt;
}

void Camera::setUpVector(atlas::math::Vector const& up)
{
    mUp = up;
}

void Camera::computeUVW()
{
    mW = glm::normalize(mEye - mLookAt);
    mU = glm::normalize(glm::cross(mUp, mW));
    mV = glm::cross(mW, mU);

    if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
        mEye.y > mLookAt.y)
    {
        mU = { 0.0f, 0.0f, 1.0f };
        mV = { 1.0f, 0.0f, 0.0f };
        mW = { 0.0f, 1.0f, 0.0f };
    }

    if (areEqual(mEye.x, mLookAt.x) && areEqual(mEye.z, mLookAt.z) &&
        mEye.y < mLookAt.y)
    {
        mU = { 1.0f, 0.0f, 0.0f };
        mV = { 0.0f, 0.0f, 1.0f };
        mW = { 0.0f, -1.0f, 0.0f };
    }
}
Fisheye::Fisheye() : Camera{}, mPsi_max{ 90.0f }
{}
void Fisheye::setPsi_max(float psi_max)
{
    mPsi_max = psi_max;
}
atlas::math::Vector Fisheye::rayDirection(atlas::math::Point const& p, ShadeRec& sr, float& r_squared) const
{
    atlas::math::Point pn{ 2.0f / sr.world->height * p.x, 2.0f / sr.world->width * p.y, 0 };
    r_squared = pn.x * pn.x + pn.y * pn.y;
    constexpr float PI_ON_180 = glm::pi<float>() / 180.0f;

    if (r_squared <= 1.0f)
    {
        float r = sqrt(r_squared);
        float psi = r * mPsi_max * PI_ON_180;
        float sin_psi = glm::sin(psi);
        float cos_psi = glm::cos(psi);
        float sin_alpha = pn.y / r;
        float cos_alpha = pn.x / r;
        atlas::math::Vector dir = sin_psi * cos_alpha * mU + sin_psi * sin_alpha * mV - cos_psi * mW;

        return dir;
    }
    else {
        return atlas::math::Vector{ 0.0f, 0.0f, 0.0f };
    }
}
void Fisheye::renderScene(std::shared_ptr<World>& world) const
{
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    int num_pixels = 0;
    float r_squared;
    Point samplePoint{}, pixelPoint{};
    Ray<atlas::math::Vector> ray{};

    ray.o = mEye;
    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ 0 }; r < world->height; ++r)
    {
        for (int c{ 0 }; c < world->width; ++c)
        {
            Colour pixelAverage{ 0, 0, 0 };

            for (int j = 0; j < world->sampler->getNumSamples(); ++j)
            {
                ShadeRec trace_data{};
                trace_data.world = world;
                trace_data.t = std::numeric_limits<float>::max();
                samplePoint = world->sampler->sampleUnitSquare();

                pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
                pixelPoint.y = r - 0.5f * world->height + samplePoint.y;

                ray.d = rayDirection(pixelPoint, trace_data, r_squared);
                bool hit{};

                if (r_squared <= 1.0f) {
                    for (auto obj : world->scene)
                    {
                        hit |= obj->hit(ray, trace_data);
                    }

                    if (hit)
                    {
                        pixelAverage += trace_data.material->areaLightShade(trace_data);
                    }
                }
            }
            pixelAverage = pixelAverage * avg;
            if (pixelAverage.r > 1.0f || pixelAverage.g > 1.0f || pixelAverage.b > 1.0f) {
                float maximum = pixelAverage.r;
                if (pixelAverage.g > maximum) {
                    maximum = pixelAverage.g;
                }
                if (pixelAverage.b > maximum) {
                    maximum = pixelAverage.b;
                }
                pixelAverage = pixelAverage / maximum;
            }
            world->image.push_back(pixelAverage);
            num_pixels++;
            printf("fisheye num pixels: %d\n", num_pixels);
        }
    }
}
Pinhole::Pinhole() : Camera{}, mDistance{ 500.0f }, mZoom{ 1.0f }
{}

void Pinhole::setDistance(float distance)
{
    mDistance = distance;
}

void Pinhole::setZoom(float zoom)
{
    mZoom = zoom;
}

atlas::math::Vector Pinhole::rayDirection(atlas::math::Point const& p) const
{
    const auto dir = p.x * mU + p.y * mV - mDistance * mW;
    return glm::normalize(dir);
}

void Pinhole::renderScene(std::shared_ptr<World>& world) const
{
    int num_pixels = 0;

    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    Point samplePoint{}, pixelPoint{};
    Ray<atlas::math::Vector> ray{};

    ray.o = mEye;
    float avg{ 1.0f / world->sampler->getNumSamples() };

    for (int r{ 0 }; r < world->height; ++r)
    {
        for (int c{ 0 }; c < world->width; ++c)
        {
            Colour pixelAverage{ 0, 0, 0 };

            for (int j = 0; j < world->sampler->getNumSamples(); ++j)
            {
                ShadeRec trace_data{};
                trace_data.world = world;
                trace_data.t = std::numeric_limits<float>::max();
                samplePoint = world->sampler->sampleUnitSquare();

                pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
                pixelPoint.y = r - 0.5f * world->height + samplePoint.y;

                ray.d = rayDirection(pixelPoint);
                bool hit{};

                for (auto obj : world->scene)
                {
                    hit |= obj->hit(ray, trace_data);
                }

                if (hit)
                {
                    pixelAverage += trace_data.material->areaLightShade(trace_data);
                }
            }
            pixelAverage = pixelAverage * avg;
            if (pixelAverage.r > 1.0f || pixelAverage.g > 1.0f || pixelAverage.b > 1.0f) {
                float maximum = pixelAverage.r;
                if (pixelAverage.g > maximum) {
                    maximum = pixelAverage.g;
                }
                if (pixelAverage.b > maximum) {
                    maximum = pixelAverage.b;
                }
                pixelAverage = pixelAverage / maximum;
            }
            world->image.push_back(pixelAverage);
            num_pixels++;
            printf("pinhole num pixels: %d\n", num_pixels);
        }
    }
}
// ***** Shape function members *****
Shape::Shape() : mColour{ 0, 0, 0 }, normal{ 0, 0, 0 }
{}

void Shape::setColour(Colour const& col)
{
    mColour = col;
}

Colour Shape::getColour() const
{
    return mColour;
}

void Shape::setMaterial(std::shared_ptr<Material> const& material)
{
    mMaterial = material;
}

std::shared_ptr<Material> Shape::getMaterial() const
{
    return mMaterial;
}
void Shape::setSampler([[maybe_unused]] std::shared_ptr<Sampler> sampler) {
    sampler_ptr = sampler;
}
atlas::math::Point Shape::sample() {
    return { 0,0, 0 };
}

// ***** Rectangle function members *****
Rectangle::Rectangle() :Shape{}, mPo{ 0, 0, 0 }, mA{ 0, 0, 0 }, mB{ 0, 0, 0 }
{}
Rectangle::Rectangle(atlas::math::Point const& po,
    atlas::math::Vector const& a,
    atlas::math::Vector const& b) : Shape{}
{
    setPo(po);
    setA(a);
    setB(b);
    normal = glm::normalize(glm::cross(mA, mB));
}
void Rectangle::setPo(atlas::math::Point const& po)
{
    mPo = po;
}
void Rectangle::setA(atlas::math::Vector const& a)
{
    mA = a;
}
void Rectangle::setB(atlas::math::Vector const& b)
{
    mB = b;
}
bool Rectangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.normal = normal;
        if (glm::dot(sr.normal, ray.d) > 0.0f) {
            sr.normal = -sr.normal;
        }
        sr.ray = ray;
        sr.color = mColour;
        sr.t = t;
        sr.hit_point = ray.o + t * ray.d;
        sr.material = mMaterial;
    }

    return intersect;
}
bool Rectangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    float t = glm::dot((mPo - ray.o), normal) / glm::dot(ray.d, normal);

    if (t <= kEpsilon) { return false; }

    atlas::math::Point p = ray.o + t * ray.d;
    atlas::math::Vector d = p - mPo;

    float ddota = glm::dot(d, mA);
    if (ddota < 0.0f || ddota > glm::dot(mA, mA)) {
        return false;
    }
    float ddotb = glm::dot(d, mB);
    if (ddotb < 0.0f || ddotb > glm::dot(mB, mB)) {
        return false;
    }

    tMin = t;
    return true;
}
atlas::math::Point Rectangle::sample()
{
    atlas::math::Point samplePoint = sampler_ptr->sampleUnitSquare();
    return (mPo + samplePoint.x * mA + samplePoint.y * mB);
}
float Rectangle::pdf([[maybe_unused]] ShadeRec& sr) {
    return  (1.0f/ glm::dot(mA, mB));
}
atlas::math::Vector Rectangle::getNormal([[maybe_unused]] atlas::math::Point const& p)
{
    return normal;
}

// ***** Triangle function members *****
Triangle::Triangle() :Shape{}, mA{ 0, 0, 0 }, mB{ 0, 0, 0 }, mC{ 0, 0, 0 }
{}
Triangle::Triangle(atlas::math::Point const& a,
    atlas::math::Point const& b,
    atlas::math::Point const& c) : Shape{}
{
    setA(a);
    setB(b);
    setC(c);
}
void Triangle::setA(atlas::math::Point const& a)
{
    mA = a;
}
void Triangle::setB(atlas::math::Point const& b)
{
    mB = b;
}
void Triangle::setC(atlas::math::Point const& c)
{
    mC = c;
}
bool Triangle::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.normal = glm::normalize(glm::cross((mB - mA), (mC - mA)));
        if (glm::dot(sr.normal, ray.d) > 0.0f) {
            sr.normal = -sr.normal;
        }
        sr.ray = ray;
        sr.color = mColour;
        sr.t = t;
        sr.hit_point = ray.o + t * ray.d;
        sr.material = mMaterial;
    }

    return intersect;
}
bool Triangle::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    float a = mA.x - mB.x, b = mA.x - mC.x, c = ray.d.x, d = mA.x - ray.o.x;
    float e = mA.y - mB.y, f = mA.y - mC.y, g = ray.d.y, h = mA.y - ray.o.y;
    float i = mA.z - mB.z, j = mA.z - mC.z, k = ray.d.z, l = mA.z - ray.o.z;

    float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
    float q = g * i - e * k, s = e * j - f * i;

    float inv_denom = 1.0f / (a * m + b * q + c * s);

    float e1 = d * m - b * n - c * p;
    float beta = e1 * inv_denom;

    if (beta < 0.0f) {
        return false;
    }

    float r = r = e * l - h * i;
    float e2 = a * n + d * q + c * r;
    float gamma = e2 * inv_denom;

    if (gamma < 0.0f) {
        return false;
    }

    if ((beta + gamma) > 1.0f) {
        return false;
    }

    float e3 = a * p - b * r + d * s;
    float t = e3 * inv_denom;


    if (atlas::core::geq(t, kEpsilon))
    {
        tMin = t;
    }

    return true;
}
atlas::math::Point Triangle::sample()
{
    return mA;
}
float Triangle::pdf([[maybe_unused]] ShadeRec& sr) {
    return 1.0f;
}
atlas::math::Vector Triangle::getNormal([[maybe_unused]] atlas::math::Point const& p)
{
    return normal;
}

// ***** Plane function members *****
Plane::Plane(atlas::math::Vector a, atlas::math::Vector n) :Shape{},
mA{ a }, mN{ n }
{}
bool Plane::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr)const
{
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };


    if (intersect && t < sr.t) {
        if (glm::dot(mN, ray.d) > 0.0f) {
            sr.normal = -mN;
        }
        else {
            sr.normal = mN;
        }
        sr.ray = ray;
        sr.hit_point = ray.o + t * ray.d;
        sr.color = mColour;
        sr.t = t;
        sr.material = mMaterial;
    }
    return intersect;
}
bool Plane::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    float t = glm::dot((mA - ray.o), mN) / glm::dot(ray.d, mN);
    if (atlas::core::geq(t, kEpsilon)) {
        tMin = t;
        return true;
    }
    return false;
}
atlas::math::Point Plane::sample()
{
    return mA;
}
float Plane::pdf([[maybe_unused]] ShadeRec& sr) {
    return 1.0f;
}
atlas::math::Vector Plane::getNormal([[maybe_unused]] atlas::math::Point const& p)
{
    return normal;
}

// ***** Sphere function members *****
Sphere::Sphere(atlas::math::Point center, float radius) :Shape{},
mCentre{ center }, mRadius{ radius }, mRadiusSqr{ radius * radius }
{}

bool Sphere::hit(atlas::math::Ray<atlas::math::Vector> const& ray,
    ShadeRec& sr) const
{
    atlas::math::Vector tmp = ray.o - mCentre;
    float t{ std::numeric_limits<float>::max() };
    bool intersect{ intersectRay(ray, t) };

    // update ShadeRec info about new closest hit
    if (intersect && t < sr.t)
    {
        sr.normal = (tmp + t * ray.d) / mRadius;
        if (glm::dot(sr.normal, ray.d) > 0.0f) {
            sr.normal = -sr.normal;
        }
        sr.ray = ray;
        sr.color = mColour;
        sr.t = t;
        sr.hit_point = ray.o + t * ray.d;
        sr.material = mMaterial;
    }

    return intersect;
}

bool Sphere::intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
    float& tMin) const
{
    const auto tmp{ ray.o - mCentre };
    const auto a{ glm::dot(ray.d, ray.d) };
    const auto b{ 2.0f * glm::dot(ray.d, tmp) };
    const auto c{ glm::dot(tmp, tmp) - mRadiusSqr };
    const auto disc{ (b * b) - (4.0f * a * c) };

    if (atlas::core::geq(disc, 0.0f))
    {
        const float e{ std::sqrt(disc) };
        const float denom{ 2.0f * a };

        // Look at the negative root first
        float t = (-b - e) / denom;
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }

        // Now the positive root
        t = (-b + e);
        if (atlas::core::geq(t, kEpsilon))
        {
            tMin = t;
            return true;
        }
    }

    return false;
}
atlas::math::Point Sphere::sample()
{
    return mCentre;
}
float Sphere::pdf([[maybe_unused]] ShadeRec& sr) {
    return 1.0f;
}
atlas::math::Vector Sphere::getNormal([[maybe_unused]] atlas::math::Point const& p)
{
    return { 0, 0, 0 };
}



// ***** Lambertian function members *****
Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}
{}

Lambertian::Lambertian(Colour diffuseColor, float diffuseReflection) :
    mDiffuseColour{ diffuseColor }, mDiffuseReflection{ diffuseReflection }
{}

Colour
Lambertian::fn([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector const& incoming) const
{
    return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();
}
Colour
Lambertian::sample_f([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector& incoming) const
{
    return mDiffuseColour * mDiffuseReflection / glm::dot(sr.normal, incoming);
}
Colour
Lambertian::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const
{
    return mDiffuseColour * mDiffuseReflection;
}

void Lambertian::setDiffuseReflection(float kd)
{
    mDiffuseReflection = kd;
}

void Lambertian::setDiffuseColour(Colour const& colour)
{
    mDiffuseColour = colour;
}
// ******* Driver Code *******
GlossySpecular::GlossySpecular() : mSpecularColour{}, mDiffuseReflection{}, mSpecularReflection{}, mExponent{}
{}
GlossySpecular::GlossySpecular(Colour specularColour, float diffuseReflection,
    float specularReflection, float exponent) :

    mSpecularColour{ specularColour }, mDiffuseReflection{ diffuseReflection },
    mSpecularReflection{ specularReflection }, mExponent{ exponent }
{}
Colour GlossySpecular::fn([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector const& incoming) const
{
    atlas::math::Vector r = glm::normalize(-incoming + 2.0f * sr.normal * (glm::dot(sr.normal, incoming)));
    float rdotwo = glm::dot(r, reflected);
    if (rdotwo > 0.0f) {
        return mSpecularReflection * glm::pow(rdotwo, mExponent) * mSpecularColour;
    }
    return sr.world->background;
}
Colour GlossySpecular::sample_f([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected,
    [[maybe_unused]] atlas::math::Vector& incoming) const
{

    atlas::math::Vector r = glm::normalize(-incoming + 2.0f * sr.normal * (glm::dot(sr.normal, -incoming)));
    float rdotwo = glm::dot(r, reflected);
    if (rdotwo > 0.0f) {
        return mSpecularReflection * glm::pow(rdotwo, mExponent) * mSpecularColour;
    }
    return sr.world->background;
}
Colour GlossySpecular::rho([[maybe_unused]] ShadeRec const& sr,
    [[maybe_unused]] atlas::math::Vector const& reflected) const
{
    return sr.world->background;
}
void GlossySpecular::setDiffuseReflection(float kd)
{
    mDiffuseReflection = kd;
}
void GlossySpecular::setSpecularReflection(float ks)
{
    mSpecularReflection = ks;
}
void GlossySpecular::setSpecularColour(Colour const& colour)
{
    mSpecularColour = colour;
}
// ***** Emissive function members *****
Emissive::Emissive() :Material{}, mColour{ Colour{0, 0, 0} }, mRadiance{ 0.0f }
{}
Emissive::Emissive(float radiance, Colour color) :
    Material{},
    mRadiance{ radiance },
    mColour{ color }
{}
Colour Emissive::L([[maybe_unused]] ShadeRec& sr)
{
    return mRadiance * mColour;
}
void Emissive::scaleRadiance(float b)
{
    mRadiance = b;
}
void Emissive::setColour(Colour const& c)
{
    mColour = c;
}
Colour Emissive::shade(ShadeRec& sr)
{
    if (glm::dot(-sr.normal, sr.ray.d) > 0.0f)
    {
        return mRadiance * mColour;
    }
    else
    {
        return { 0, 0, 0 };
    }
}
Colour Emissive::areaLightShade(ShadeRec& sr)
{
    if (glm::dot(-sr.normal, sr.ray.d) > 0.0f)
    {
        return mRadiance * mColour;
    }
    return { 0, 0, 0 };
}
// ***** Phong function members *****
Phong::Phong() :
    Material{},
    mAmbientBRDF{ std::make_shared<Lambertian>() },
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mSpecularBRDF{ std::make_shared<GlossySpecular>() },
    mExponent{ 0.0f }
{}
Phong::Phong(float ks, float kd, float ka, Colour color, float exponent) : Phong{}
{
    setAmbientReflection(ka);
    setDiffuseReflection(kd);
    setSpecularReflection(ks);
    setSpecularColour(color);
    setExponent(exponent);
}
void Phong::setExponent(float k)
{
    mExponent = k;
}
void Phong::setAmbientReflection(float k)
{
    mAmbientBRDF->setDiffuseReflection(k);
}
void Phong::setDiffuseReflection(float k)
{
    mDiffuseBRDF->setDiffuseReflection(k);
}
void Phong::setSpecularReflection(float ks)
{
    mSpecularBRDF->setSpecularReflection(ks);
}
void Phong::setSpecularColour(Colour colour)
{
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
    mSpecularBRDF->setSpecularColour(colour);
}
Colour Phong::L([[maybe_unused]] ShadeRec& sr)
{
    return { 0, 0, 0 };
}
Colour Phong::shade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.d;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            bool in_shadow = false;
            Ray shadowRay(sr.hit_point, wi);
            in_shadow = sr.world->lights[i]->in_shadow(shadowRay, sr);

            if (!in_shadow) {
                L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi))
                    * sr.world->lights[i]->L(sr) * nDotWi;
            }
        }

    }

    return L;
}
Colour Phong::areaLightShade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.d;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            bool in_shadow = false;
            Ray shadowRay(sr.hit_point, wi);
            in_shadow = sr.world->lights[i]->in_shadow(shadowRay, sr);

            if (!in_shadow) {
                L += ( (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) * sr.world->lights[i]->L(sr)
                    * sr.world->lights[i]->G(sr) * nDotWi / sr.world->lights[i]->pdf(sr) );
            }
        }
    }

    return L;
}
// ***** Matte function members *****
Matte::Matte() :
    Material{},
    mDiffuseBRDF{ std::make_shared<Lambertian>() },
    mAmbientBRDF{ std::make_shared<Lambertian>() }
{}

Matte::Matte(float kd, float ka, Colour color) : Matte{}
{
    setDiffuseReflection(kd);
    setAmbientReflection(ka);
    setDiffuseColour(color);
}

void Matte::setDiffuseReflection(float k)
{
    mDiffuseBRDF->setDiffuseReflection(k);
}

void Matte::setAmbientReflection(float k)
{
    mAmbientBRDF->setDiffuseReflection(k);
}

void Matte::setDiffuseColour(Colour colour)
{
    mDiffuseBRDF->setDiffuseColour(colour);
    mAmbientBRDF->setDiffuseColour(colour);
}

Colour Matte::L([[maybe_unused]] ShadeRec& sr)
{
    return { 0, 0, 0 };
}
Colour Matte::shade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.d;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();

    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            L += mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr) *
                nDotWi;
        }
    }

    return L;
}
Colour Matte::areaLightShade(ShadeRec& sr)
{
    using atlas::math::Ray;
    using atlas::math::Vector;

    Vector wo = -sr.ray.d;
    Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);
    size_t numLights = sr.world->lights.size();
    for (size_t i{ 0 }; i < numLights; ++i)
    {
        Vector wi = sr.world->lights[i]->getDirection(sr);
        float nDotWi = glm::dot(sr.normal, wi);

        if (nDotWi > 0.0f)
        {
            bool in_shadow = false;
            Ray shadowRay(sr.hit_point, wi);
            in_shadow = sr.world->lights[i]->in_shadow(shadowRay, sr);

            if (!in_shadow) {
                L += (mDiffuseBRDF->fn(sr, wo, wi) * sr.world->lights[i]->L(sr)
                    * sr.world->lights[i]->G(sr) * nDotWi / sr.world->lights[i]->pdf(sr));
            }
        }
    }
    return L;
}


// ***** Light function members *****
Light::Light() : mColour{ Colour{0, 0, 0} }, mRadiance{ 0.0f }
{}
void Light::scaleRadiance(float b)
{
    mRadiance = b;
}

void Light::setColour(Colour const& c)
{
    mColour = c;
}

// ***** AreaLight function members *****
AreaLight::AreaLight() :Light{}, samplePoint{ atlas::math::Point{0, 0, 0} }, normal{ 0, 0, 0 }, wi{ 0, 0, 0 }
{}
AreaLight::AreaLight(std::shared_ptr<Shape> const& object, std::shared_ptr<Material> const& material) : Light{}
{
    mObject = object;
    mMaterial = material;
    normal = { 0, 0, 0 };
    wi = { 0, 0, 0 };
}
void AreaLight::setObject(std::shared_ptr<Shape> const& object)
{
    mObject = object;
}
void AreaLight::setMaterial(std::shared_ptr<Material> const& material)
{
    mMaterial = material;
}
bool AreaLight::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{
    float t;
    std::size_t num_objects = sr.world->scene.size();
    float ts = glm::dot((samplePoint - ray.o), ray.d);
    for (int i = 0; i < num_objects; ++i)
    {
        if (sr.world->scene[i]->intersectRay(ray, t) && t < ts) {
            return true;
        }

    }
    return false;
}
atlas::math::Vector AreaLight::getDirection(ShadeRec& sr)
{
    samplePoint = mObject->sample();
    normal = mObject->getNormal(samplePoint);
    wi = glm::normalize(samplePoint - sr.hit_point);

    return wi;
}
Colour AreaLight::L(ShadeRec& sr)const
{
    float ndotd = glm::dot(-normal, wi);
    if (ndotd > 0.0f) {
        return mMaterial->L(sr);
    }
    else {
        return { 0, 0, 0 };
    }
}
float AreaLight::G([[maybe_unused]] ShadeRec& sr)
{
    
    float ndotd = glm::dot(-normal, wi);
    atlas::math::Point hitpoint= sr.hit_point;
    float d2 =  (samplePoint.x - hitpoint.x)* (samplePoint.x - hitpoint.x)
        + (samplePoint.y - hitpoint.y) * (samplePoint.y - hitpoint.y)
        + (samplePoint.z - hitpoint.z) * (samplePoint.z - hitpoint.z);
    
    return  (ndotd / d2);
}
float AreaLight::pdf([[maybe_unused]] ShadeRec& sr) {
    
    return mObject->pdf(sr);
}

// ***** PointLight function members *****
PointLight::PointLight() : Light{}
{
    mLocation = { 0, 0, 0 };
}
PointLight::PointLight(atlas::math::Point const& location) : Light{} {
    setLocation(location);
}
void PointLight::setLocation(atlas::math::Point const& location)
{
    mLocation = location;
}
atlas::math::Vector PointLight::getDirection(ShadeRec& sr)
{
    return glm::normalize(mLocation - sr.hit_point);
}
bool PointLight::in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
    [[maybe_unused]] ShadeRec& sr)const
{

    float t;
    std::size_t num_objects = sr.world->scene.size();
    float d = glm::length(mLocation - ray.o);

    for (int i = 0; i < num_objects; ++i) {
        if (sr.world->scene[i]->intersectRay(ray, t) && t < d) {
            return true;
        }
    }

    return false;
}
Colour PointLight::L([[maybe_unused]] ShadeRec& sr)const
{
    return mRadiance * mColour;
}
float PointLight::G([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}
float PointLight::pdf([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}

// ***** Directional function members *****
Directional::Directional() : Light{}
{
    mDirection = { 0, 0, 0 };
}

Directional::Directional(atlas::math::Vector const& d) : Light{}
{
    setDirection(d);
}
void Directional::setDirection(atlas::math::Vector const& d)
{
    mDirection = glm::normalize(d);
}

atlas::math::Vector Directional::getDirection([[maybe_unused]] ShadeRec& sr)
{
    return mDirection;
}
bool Directional::in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
    [[maybe_unused]] ShadeRec& sr)const
{
    float t;
    std::size_t num_objects = sr.world->scene.size();

    for (int i = 0; i < num_objects; ++i) {
        if (sr.world->scene[i]->intersectRay(ray, t) && atlas::core::geq(t, kEpsilon)) {
            return true;
        }
    }
    return false;
}
Colour Directional::L([[maybe_unused]] ShadeRec& sr)const
{
    return mRadiance * mColour;
}
float Directional::G([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}
float Directional::pdf([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}
// ***** Ambient function members *****
Ambient::Ambient() : Light{}, mDirection{ 0, 0, 0 }
{}

atlas::math::Vector Ambient::getDirection([[maybe_unused]] ShadeRec& sr)
{
    return atlas::math::Vector{ 0.0f };
}
bool Ambient::in_shadow([[maybe_unused]] atlas::math::Ray<atlas::math::Vector> const& ray,
    [[maybe_unused]] ShadeRec& sr)const
{
    return false;
}
Colour Ambient::L([[maybe_unused]] ShadeRec& sr)const
{
    return mRadiance * mColour;
}
float Ambient::G([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}
float Ambient::pdf([[maybe_unused]] ShadeRec& sr)
{
    return 1.0f;
}


int main()
{
    using atlas::math::Point;
    using atlas::math::Ray;
    using atlas::math::Vector;

    std::shared_ptr<World> world{ std::make_shared<World>() };

    world->width = 600;
    world->height = 600;
    world->background = { 0, 0, 0 };
    world->sampler = std::make_shared<Jittered>(100, 80);


    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 0, 0, -600 }, 100.0f));
    world->scene[0]->setMaterial(
        std::make_shared<Phong>(0.05f, 0.50f, 0.05f, Colour{ 1, 0, 0 }, 0.4f));
    world->scene[0]->setColour({ 1, 0, 0 });

    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 128, 32, -700 }, 64.0f));
    world->scene[1]->setMaterial(
        std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0, 0, 1 }));
    world->scene[1]->setColour({ 0, 0, 1 });

    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ -128, 32, -700 }, 64.0f));
    world->scene[2]->setMaterial(
        std::make_shared<Matte>(0.50f, 0.05f, Colour{ 0, 1, 0 }));
    world->scene[2]->setColour({ 0, 1, 0 });


    world->scene.push_back(
        std::make_shared<Plane>(atlas::math::Vector{ 0, 200, 0 }, atlas::math::Vector{ 0,1,0 }));
    world->scene[3]->setMaterial(
        std::make_shared<Phong>(0.05f, 0.5f, 0.05f, Colour{ 1.0f, 1.0f, 1.0f }, 0.05f));
    world->scene[3]->setColour({ 1.0f, 1.0f, 1.0f });

    world->scene.push_back(
        std::make_shared<Triangle>(atlas::math::Point{ 400, 200, -400 },
            atlas::math::Point{ 400, -1000, -400 }, atlas::math::Point{ 700, 200, -400 }));
    world->scene[4]->setMaterial(
        std::make_shared<Phong>(0.6f, 0.5f, 0.05f, Colour{ 0.32f, 0.23f, 0.65f }, 0.05f));
    world->scene[4]->setColour({ 0.32f, 0.23f, 0.65f });

    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ 150, 150, -650 }, 40.0f));
    world->scene[5]->setMaterial(
        std::make_shared<Phong>(0.01f, 0.50f, 0.05f, Colour{ 0, 0.5f, 1 }, 0.05f));
    world->scene[5]->setColour({ 0, 0.5f, 1 });

    world->scene.push_back(
        std::make_shared<Sphere>(atlas::math::Point{ -170, 125, 90 }, 50.0f));
    world->scene[6]->setMaterial(
        std::make_shared<Phong>(0.05f, 0.50f, 0.05f, Colour{ 1, 0.843f, 0 }, 0.4f));
    world->scene[6]->setColour({ 1, 0.843f, 0 });


    std::shared_ptr<Rectangle> lightRectangle{ std::make_shared<Rectangle>(
       atlas::math::Point {-400, -200, -1000}, 
        atlas::math::Vector{100, 0, 0}, atlas::math::Vector{0, 100, 0}) };

    lightRectangle->setMaterial(std::make_shared<Emissive>(10.0f, Colour{ 1, 1, 1 }));
    lightRectangle->setSampler(std::make_shared<Jittered>(100, 80));
    lightRectangle->setColour(Colour{ 1, 1, 1 });

    world->scene.push_back(lightRectangle);

    world->ambient = std::make_shared<Ambient>();
    world->lights.push_back(
        std::make_shared<Directional>(Directional{ {-1, -1, 1} }));
    world->lights.push_back(
        std::make_shared<PointLight>(PointLight{ { 250, -200, -500 } }));
   
    std::shared_ptr<AreaLight> areaLight{ std::make_shared<AreaLight>() };
    areaLight->setObject(lightRectangle);
    areaLight->setMaterial(lightRectangle->getMaterial());
    world->lights.push_back(areaLight);

    world->ambient->setColour({ 1, 1, 1 });
    world->ambient->scaleRadiance(0.05f);

    world->lights[0]->setColour({ 1, 1, 1 });
    world->lights[0]->scaleRadiance(3.0f);

    world->lights[1]->setColour({ 1, 1, 1 });
    world->lights[1]->scaleRadiance(3.0f);


    Pinhole camera{};
    camera.setEye({ 0.0f, -200.0f, 400.0f });
    camera.computeUVW();
    camera.renderScene(world);
    saveToFile("raytrace_pinhole.bmp", world->width, world->height, world->image);


    Fisheye camera_fisheye{};
    camera_fisheye.setEye({ 0.0f, -200.0f, 100.0f });
    camera_fisheye.computeUVW();
    world->image.clear();
    camera_fisheye.renderScene(world);
    saveToFile("raytrace_fisheye.bmp", world->width, world->height, world->image);

    return 0;
}

void saveToFile(std::string const& filename,
    std::size_t width,
    std::size_t height,
    std::vector<Colour> const& image)
{
    std::vector<unsigned char> data(image.size() * 3);

    for (std::size_t i{ 0 }, k{ 0 }; i < image.size(); ++i, k += 3)
    {
        Colour pixel = image[i];
        data[k + 0] = static_cast<unsigned char>(pixel.r * 255);
        data[k + 1] = static_cast<unsigned char>(pixel.g * 255);
        data[k + 2] = static_cast<unsigned char>(pixel.b * 255);
    }

    stbi_write_bmp(filename.c_str(),
        static_cast<int>(width),
        static_cast<int>(height),
        3,
        data.data());
}