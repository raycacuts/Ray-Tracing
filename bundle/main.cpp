#include "assignment.hpp"
#include <atlas/utils/LoadObjFile.hpp>
#include <atlas/math/Solvers.hpp>
#include <math.h>

using Ray = atlas::math::Ray<atlas::math::Vector>;
const float kEpsilon{ 0.05f };
// ******* Function Member Implementation *******

// ***** World function members *****
World::World() : width{ 0 }, height{ 0 }, background{ Colour{0, 0, 0} }, max_depth{ 0 }
{}
World::World(std::size_t pWidth, std::size_t pHeight, Colour pBackground, int depth)
{
	width = pWidth;
	height = pHeight;
	background = pBackground;
	max_depth = depth;
}
void World::setSampler(std::shared_ptr<Sampler> const& pSampler)
{
	sampler = pSampler;
}
void World::setCamera(std::shared_ptr<Camera> const& pCamera)
{
	camera = pCamera;
}
void World::setTracer(std::shared_ptr<Tracer> const& pTracer)
{
	tracer = pTracer;
}

void World::addObject(std::shared_ptr<Shape> const& pObject)
{
	objects.push_back(pObject);
}

void World::build()
{
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	//reflective red
	objects.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 0, 0, -500 }, 100.0f));
	std::shared_ptr<Reflective> reflective{ std::make_shared<Reflective>(0.6f, Colour{ 0.85, 0, 0 }) };
	reflective->setDiffuseReflection(0.1f);
	reflective->setAmbientReflection(0.1f);
	reflective->setSpecularReflection(0.1f);
	reflective->setSampler(std::make_shared<Jittered>(9, 10));
	reflective->setExponent(100);

	objects[0]->setMaterial(reflective);
	objects[0]->setColour({ 0.85, 0, 0 });

	// glossy green sphere
	objects.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ -250, 32, -500 }, 100.0f));
	std::shared_ptr<GlossyReflector> glossy{ std::make_shared<GlossyReflector>(0.9f,
		Colour{ 0, 0.85, 0 }) };
	glossy->setKd(0.1f);
	glossy->setKa(0.1f);
	glossy->setKs(0.1f);
	glossy->setExponent(10);
	glossy->setSampler(std::make_shared<Jittered>(9, 10));
	objects[1]->setMaterial(glossy);
	objects[1]->setColour({ 0, 0.85, 0 });


	//  floor plane
	objects.push_back(
		std::make_shared<Plane>(atlas::math::Vector{ 0, 200, 0 }, atlas::math::Vector{ 0,1,0 }));
	std::shared_ptr<Reflective> reflective_plane{ std::make_shared<Reflective>(0.01f,
														Colour{ 1.0f, 1.0f, 1.0f }) };
	reflective_plane->setDiffuseReflection(0.1f);
	reflective_plane->setAmbientReflection(0.1f);
	reflective_plane->setSpecularReflection(0.2f);
	reflective_plane->setSampler(std::make_shared<Jittered>(4, 10));
	reflective_plane->setExponent(100);
	objects[2]->setMaterial(reflective_plane);
	objects[2]->setColour({ 0.5f, 0.5f, 0.5f });

	//triangle
	objects.push_back(
		std::make_shared<Triangle>(atlas::math::Point{ 300, 200, -600 },
			atlas::math::Point{ 400, -1000, -400 }, atlas::math::Point{ 700, 200, -400 }));
	std::shared_ptr<Transparent> tri_glass{ std::make_shared<Transparent>() };
	tri_glass->setAmbientReflection(0.1f);
	tri_glass->setDiffuseReflection(0.0f);
	tri_glass->setSpecularReflection(0.0f);
	tri_glass->setExponent(1000.0f);
	tri_glass->setKr(0.75f);
	tri_glass->setKt(0.9f);
	tri_glass->setIor(1.1f);
	tri_glass->setColour({ 0.9, 1, 0.9 });
	tri_glass->setSampler(std::make_shared<Jittered>(4, 10));
	objects[3]->setMaterial(tri_glass);
	objects[3]->setColour({ 0.32f, 0.23f, 0.65f });

	//reflective behind
	objects.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 500, -100, -800 }, 50.0f));
	std::shared_ptr<Reflective> reflective_behind{ std::make_shared<Reflective>(0.75f, Colour{ 0.85, 0, 0 }) };
	reflective_behind->setDiffuseReflection(0.1f);
	reflective_behind->setAmbientReflection(0.1f);
	reflective_behind->setSpecularReflection(0.1f);
	reflective_behind->setSampler(std::make_shared<Jittered>(4, 10));
	reflective_behind->setExponent(100);

	objects[4]->setMaterial(reflective_behind);
	objects[4]->setColour({ 0.85, 0, 0 });

	//realistic glass blue
	objects.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 150, 120, -500 }, 50.0f));
	std::shared_ptr<Dielectric> glass{ std::make_shared<Dielectric>() };
	glass->setAmbientReflection(0.01f);
	glass->setDiffuseReflection(0.0f);
	glass->setSpecularReflection(0.0f);
	glass->setExponent(2000.0f);
	glass->setKr(0.01f);
	glass->setEtaIn(1.5f);
	glass->setEtaOut(1.0f);
	glass->setColourIn({ 0, 0.5, 1 });
	glass->setColourOut({ 0, 0.5, 1 });
	glass->setSampler(std::make_shared<Jittered>(4, 10));

	objects[5]->setMaterial(glass);
	objects[5]->setColour({ 0, 0.5, 1 });

	//glass white
	objects.push_back(
		std::make_shared<Sphere>(atlas::math::Point{ 200, -120, -500 }, 80.0f));
	std::shared_ptr<Transparent> glass_red{ std::make_shared<Transparent>() };
	glass_red->setAmbientReflection(0.0f);
	glass_red->setDiffuseReflection(0.0f);
	glass_red->setSpecularReflection(0.0f);
	glass_red->setExponent(2000.0f);
	glass_red->setKr(0.0f);
	glass_red->setKt(0.9f);
	glass_red->setIor(1.1f);
	glass_red->setColour(Colour{ 0.9, 1.0, 1.0 });
	glass_red->setSampler(std::make_shared<Jittered>(4, 10));

	objects[6]->setMaterial(glass_red);
	objects[6]->setColour({ 0.9, 1.0, 1.0 });


	//mirror rectangle

	std::shared_ptr<Rectangle> rectangle{ std::make_shared<Rectangle>(
	   atlas::math::Point {-500, 150, -900}, atlas::math::Vector{1500, 0, 0},
		atlas::math::Vector{0, -1000, 0}) };
	objects.push_back(rectangle);

	std::shared_ptr<Reflective> reflective_mirror_plane{ std::make_shared<Reflective>(0.1f, Colour{1, 0.843f, 0 }) };
	reflective_mirror_plane->setDiffuseReflection(0.1f);
	reflective_mirror_plane->setAmbientReflection(0.1f);
	reflective_mirror_plane->setSpecularReflection(0.1f);
	reflective_mirror_plane->setExponent(100);
	reflective_mirror_plane->setSampler(std::make_shared<Jittered>(4, 10));

	objects[7]->setMaterial(reflective_mirror_plane);
	objects[7]->setColour({ 1, 0.843f, 0 });

	/*
	std::vector<atlas::utils::Shape> mesh_shapes = atlas::utils::loadObjMesh("dragon.obj")->shapes;
	std::vector<Triangle> mesh;

	for (std::vector<atlas::utils::Shape>::size_type i = 0; i < mesh_shapes.size(); i++) {
		for (std::size_t j = 0; (j + 3) < mesh_shapes[i].indices.size(); j += 3) {
			mesh.push_back(Triangle{ mesh_shapes[i].vertices[mesh_shapes[i].indices[j]],
				mesh_shapes[i].vertices[mesh_shapes[i].indices[j + 1]],
				mesh_shapes[i].vertices[mesh_shapes[i].indices[j + 2]] });
		}
	}
	Phong meshMaterial{ 0.05f, 0.50f, 0.05f, Colour{ 1, 0.843f, 0 }, 0.4f };
	for (int i{ 0 }; i < mesh.size(); ++i) {
		world->scene.push_back(std::make_shared<Triangle>(mesh[i]));
		world->scene[i + 7]->setMaterial(std::make_shared<Phong>(meshMaterial));
		world->scene[i + 7]->setColour({ 1, 0.843f, 0 });

	}
	*/



	//ambient occulader
	std::shared_ptr<Jittered> ambientSampler{ std::make_shared<Jittered>(36, 80) };
	std::shared_ptr<AmbientOccluder> ambientLight{ std::make_shared<AmbientOccluder>() };
	ambientLight->setMinAmount(0.5f);
	ambientLight->setSampler(ambientSampler);

	ambient = ambientLight;// std::make_shared<Ambient>();// ambientLight;
	ambient->setColour({ 1, 1, 1 });
	ambient->scaleRadiance(5.0f);

	//point light
	lights.push_back(std::make_shared<PointLight>(PointLight{ { 250, -200, -400 } }));
	lights[0]->setColour({ 1, 1, 1 });
	lights[0]->scaleRadiance(1.0f);

	//area light
	std::shared_ptr<Rectangle> lightRectangle{ std::make_shared<Rectangle>(
	   atlas::math::Point {-500, -200, -600}, atlas::math::Vector{100, 0, 0}, atlas::math::Vector{0, 100, 0}) };

	lightRectangle->setMaterial(std::make_shared<Emissive>(500.0f, Colour{ 1, 1, 1 }));
	lightRectangle->setSampler(std::make_shared<Jittered>(25, 80));
	lightRectangle->setColour({ 1, 1, 1 });

	objects.push_back(lightRectangle);

	std::shared_ptr<AreaLight> areaLight{ std::make_shared<AreaLight>() };
	areaLight->setObject(lightRectangle);
	areaLight->setMaterial(lightRectangle->getMaterial());
	lights.push_back(areaLight);

}
ShadeRec World::hit_objects(const Ray& ray)
{
	ShadeRec sr;
	sr.hit_object = false;
	sr.tmin = std::numeric_limits<float>::max();
	sr.color = Colour{ 0, 0, 0 };
	std::size_t numObjects = objects.size();

	for (int i{ 0 }; i < numObjects; i++)
	{
		objects[i]->hit(ray, sr);

	}
	return sr;
}

// ***** Tracer *****
Tracer::Tracer() : world{ std::make_shared<World>() }
{}
Tracer::Tracer(std::shared_ptr<World> pWorld)
{
	world = pWorld;
}
Colour Tracer::trace_ray([[maybe_unused]] const Ray& ray) const
{
	return Colour{ 0, 0, 0 };
}
Colour Tracer::trace_ray([[maybe_unused]] const Ray ray, [[maybe_unused]] const int depth) const
{
	return Colour{ 0, 0, 0 };
}
Colour Tracer::trace_ray([[maybe_unused]] const Ray ray, [[maybe_unused]] float& tmin, [[maybe_unused]] const int depth) const
{
	return Colour{ 0, 0, 0 };
}
// ***** RayCast funtion members *****
PathTrace::PathTrace() : Tracer()
{}
PathTrace::PathTrace(std::shared_ptr<World> pWorld) : Tracer(pWorld)
{}
Colour PathTrace::trace_ray(const Ray& ray) const
{
	ShadeRec sr(world->hit_objects(ray));
	sr.world = world;

	if (sr.hit_object) {
		sr.ray = ray;
		return sr.material->pathShade(sr);
	}
	else {
		return world->background;
	}
}
Colour PathTrace::trace_ray(const Ray ray, [[maybe_unused]] const int depth) const
{
	if (depth >= world->max_depth) {
		return Colour{ 0, 0, 0 };
	}
	else
	{
		ShadeRec sr{ world->hit_objects(ray) };
		sr.world = world;

		if (sr.hit_object) {
			sr.ray = ray;
			sr.depth = depth;
			return sr.material->pathShade(sr);
		}
		else {
			return world->background;
		}
	}

}
Colour PathTrace::trace_ray(const Ray ray, float& tmin, const int depth) const
{
	if (depth >= world->max_depth) {
		tmin = std::numeric_limits<float>::max();
		return Colour{ 0, 0, 0 };
	}
	else
	{
		ShadeRec sr{ world->hit_objects(ray) };
		sr.world = world;

		if (sr.hit_object) {
			sr.ray = ray;
			sr.depth = depth;
			tmin = sr.tmin;
			return sr.material->pathShade(sr);
		}
		else {
			tmin = std::numeric_limits<float>::max();
			return world->background;
		}
	}
}
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

void Sampler::mapSamplesToHemisphere(const float e)
{
	std::size_t size = mSamples.size();
	mHemisphereSamples.reserve(mNumSets * mNumSamples);

	double pi = std::acos(-1);

	for (std::size_t j{ 0 }; j < size; j++) {
		double cos_phi = cos(2.0 * pi * mSamples[j].x);
		double sin_phi = sin(2.0 * pi * mSamples[j].x);
		double cos_theta = std::pow((1.0 - mSamples[j].y), 1.0 / (e + 1.0));
		double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
		double pu = sin_theta * cos_phi;
		double pv = sin_theta * sin_phi;
		double pw = cos_theta;

		mHemisphereSamples.push_back(atlas::math::Point(pu, pv, pw));
	}
}
atlas::math::Point Sampler::sampleHemiSphere() {
	if (mCount % mNumSamples == 0)
	{
		atlas::math::Random<int> engine;
		mJump = (engine.getRandomMax() % mNumSets) * mNumSamples;
	}
	return mHemisphereSamples[(mJump + mShuffledIndeces[mJump + mCount++ % mNumSamples]) % mHemisphereSamples.size()];
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
	mLookAt{ 0.0f, 0.0f, -250.0f },
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

void Camera::setWorld(std::shared_ptr<World>& pWorld)
{
	world = pWorld;
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

void Pinhole::renderScene() const
{
	int num_pixels = 0;
	int num = 0;
	using atlas::math::Point;
	using atlas::math::Ray;
	using atlas::math::Vector;

	Point samplePoint{}, pixelPoint{};
	Ray<atlas::math::Vector> ray{};

	//int depth = 0;
	ray.o = mEye;

	float avg{ 1.0f / world->sampler->getNumSamples() };

	for (int r{ 0 }; r < world->height; ++r)
	{
		for (int c{ 0 }; c < world->width; ++c)
		{
			Colour pixelAverage{ 0, 0, 0 };

			for (int j = 0; j < world->sampler->getNumSamples(); ++j)
			{
				samplePoint = world->sampler->sampleUnitSquare();

				pixelPoint.x = c - 0.5f * world->width + samplePoint.x;
				pixelPoint.y = r - 0.5f * world->height + samplePoint.y;

				ray.d = rayDirection(pixelPoint);


				pixelAverage += world->tracer->trace_ray(ray, 0);


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
			if (num_pixels - num == 1000) {
				num = num_pixels;
				printf("pinhole num pixels: %d\n", num);
			}
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

std::shared_ptr<Material> Shape::getMaterial()
{
	return mMaterial;
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
	ShadeRec& sr) 
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.tmin)
	{
		sr.hit_object = true;
		sr.normal = normal;
		if (glm::dot(sr.normal, ray.d) > 0.0f) {
			sr.normal = -sr.normal;
		}
		sr.ray = ray;
		sr.color = mColour;
		sr.tmin = t;
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
	return  (1.0f / (glm::length(mA) * glm::length(mB)));
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
Triangle::Triangle(atlas::utils::Vertex const& x, atlas::utils::Vertex const& y,
	atlas::utils::Vertex const& z) : Shape{}
{
	setA(x.position);
	setB(y.position);
	setC(z.position);
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
	ShadeRec& sr) 
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.tmin)
	{
		sr.hit_object = true;
		sr.normal = glm::normalize(glm::cross((mB - mA), (mC - mA)));
		if (glm::dot(sr.normal, ray.d) > 0.0f) {
			sr.normal = -sr.normal;
		}
		sr.ray = ray;
		normal = sr.normal;
		sr.color = mColour;
		sr.tmin = t;
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
	ShadeRec& sr)
{
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };


	if (intersect && t < sr.tmin) {
		sr.hit_object = true;
		if (glm::dot(mN, ray.d) > 0.0f) {
			sr.normal = -mN;
		}
		else {
			sr.normal = mN;
		}
		normal = sr.normal;
		sr.ray = ray;
		sr.hit_point = ray.o + t * ray.d;
		sr.color = mColour;
		sr.tmin = t;
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
	ShadeRec& sr) 
{
	atlas::math::Vector tmp = ray.o - mCentre;
	float t{ std::numeric_limits<float>::max() };
	bool intersect{ intersectRay(ray, t) };

	// update ShadeRec info about new closest hit
	if (intersect && t < sr.tmin)
	{
		sr.hit_object = true;
		sr.normal = (tmp + t * ray.d) / mRadius;
		if (glm::dot(sr.normal, ray.d) > 0.0f) {
			sr.normal = -sr.normal;
		}
		normal = sr.normal;
		sr.ray = ray;
		sr.color = mColour;
		sr.tmin = t;
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

//***** FresnelTransmitter function members *****
FresnelTransmitter::FresnelTransmitter() : mEtaIn{ 0.0f }, mEtaOut{ 0.0 }
{}
FresnelTransmitter::FresnelTransmitter(float etain, float etaout) : FresnelTransmitter()
{
	mEtaIn = etain;
	mEtaOut = etaout;
}
bool FresnelTransmitter::tir(const ShadeRec& sr) const {
	atlas::math::Vector wo = -sr.ray.d;
	float cos_thetai = glm::dot(sr.normal, wo);
	float eta;

	if (cos_thetai < 0.0f) {
		eta = mEtaOut / mEtaIn;
	}
	else {
		eta = mEtaIn / mEtaOut;
	}
	return ((1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta)) < 0.0f);
}
Colour FresnelTransmitter::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return Colour{ 0, 0, 0 };
}
Colour FresnelTransmitter::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour{ 0, 0, 0 };
}
void FresnelTransmitter::setColour(Colour color) {
	mColour = color;
}
Colour FresnelTransmitter::sample_f(ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& wt) const
{
	atlas::math::Vector n = sr.normal;
	float cos_theta_i = glm::dot(n, reflected);
	float eta;

	if (cos_theta_i < 0.0f) {
		cos_theta_i = -cos_theta_i;
		n = -n;
		eta = mEtaOut / mEtaIn;
	}
	else
	{
		eta = mEtaIn / mEtaOut;
	}
	float temp = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	float cos_theta2 = sqrt(temp);
	wt = -reflected / eta - (cos_theta2 - cos_theta_i / eta) * n;

	return (fresnel(sr) / (eta * eta) * Colour { 1, 1, 1 } / std::fabs(glm::dot(sr.normal, wt)));
}
float FresnelTransmitter::fresnel(const ShadeRec& sr) const
{
	atlas::math::Vector normal = sr.normal;
	float ndotd = glm::dot(-normal, sr.ray.d);
	float eta;

	if (ndotd < 0.0f) {
		normal = -normal;
		eta = mEtaOut / mEtaIn;
	}
	else
	{
		eta = mEtaIn / mEtaOut;
	}
	float cos_theta_i = glm::dot(-normal, sr.ray.d);
	//float temp = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	float cos_theta_t = sqrt(1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta));
	float r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
	float r_perpendicular = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
	float kr = 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);
	float kt = 1.0f - kr;

	return kt;
}
void FresnelTransmitter::setEtaIn(float etain) {
	mEtaIn = etain;
}
void FresnelTransmitter::setEtaOut(float etaout) {
	mEtaOut = etaout;
}
void FresnelTransmitter::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(100);
}

//***** PerfectTransmitter function members *****
PerfectTransmitter::PerfectTransmitter() : mIor{ 0.0f }, mKt{ 0.0 }
{}
PerfectTransmitter::PerfectTransmitter(float ior, float kt) : PerfectTransmitter()
{
	mIor = ior;
	mKt = kt;
}
bool PerfectTransmitter::tir(const ShadeRec& sr) const {
	atlas::math::Vector wo = -sr.ray.d;
	float cos_thetai = glm::dot(sr.normal, wo);
	float eta = mIor;

	if (cos_thetai < 0.0f) {
		eta = 1.0f / eta;
	}
	return ((1.0f - (1.0f - cos_thetai * cos_thetai) / (eta * eta)) < 0.0f);
}
Colour PerfectTransmitter::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return Colour{ 0, 0, 0 };
}
Colour PerfectTransmitter::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour{ 0, 0, 0 };
}
void PerfectTransmitter::setColour(Colour color) {
	mColour = color;
}
Colour PerfectTransmitter::sample_f(ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& wt) const
{
	atlas::math::Vector n = sr.normal;
	float cos_theta_i = glm::dot(n, reflected);
	float eta = mIor;

	if (cos_theta_i < 0.0f) {
		cos_theta_i = -cos_theta_i;
		n = -n;
		eta = 1.0f / eta;
	}
	float temp = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	float cos_theta2 = sqrt(temp);
	wt = -reflected / eta - (cos_theta2 - cos_theta_i / eta) * n;

	return (mKt / (eta * eta) * Colour { 1, 1, 1 } / std::fabs(glm::dot(sr.normal, wt)));
}
void PerfectTransmitter::setIor(float ior) {
	mIor = ior;
}
void PerfectTransmitter::setKt(float kt) {
	mKt = kt;
}
void PerfectTransmitter::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(1);
}

//***** FresnelTransmitter function members *****
FresnelReflector::FresnelReflector() : mEtaIn{ 0.0f }, mEtaOut{ 0.0 }
{}
FresnelReflector::FresnelReflector(float etain, float etaout) : FresnelReflector()
{
	mEtaIn = etain;
	mEtaOut = etaout;
}

Colour FresnelReflector::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const
{
	return Colour{ 0, 0, 0 };
}
Colour FresnelReflector::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const
{
	return Colour{ 0, 0, 0 };
}
void FresnelReflector::setColour(Colour color) {
	mColour = color;
}
Colour FresnelReflector::sample_f(ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& wt) const
{
	float ndotwo = glm::dot(sr.normal, reflected);
	wt = -reflected + 2.0f * sr.normal * ndotwo;

	return fresnel(sr) * Colour { 1, 1, 1 } / std::fabs(glm::dot(sr.normal, wt));
}
Colour FresnelReflector::sample_f([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& wt, [[maybe_unused]] float& pdf) const
{
	float ndotwo = glm::dot(sr.normal, reflected);
	wt = -reflected + 2.0f * sr.normal * ndotwo;

	return fresnel(sr) * Colour { 1, 1, 1 } / std::fabs(glm::dot(sr.normal, wt)); //Colour{0, 0, 0};
}
float FresnelReflector::fresnel(const ShadeRec& sr) const
{
	atlas::math::Vector normal = sr.normal;
	float ndotd = glm::dot(-normal, sr.ray.d);
	float eta;

	if (ndotd < 0.0f) {
		normal = -normal;
		eta = mEtaOut / mEtaIn;
	}
	else
	{
		eta = mEtaIn / mEtaOut;
	}
	float cos_theta_i = glm::dot(-normal, sr.ray.d);
	//float temp = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	float cos_theta_t = sqrt(1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta));
	float r_parallel = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);
	float r_perpendicular = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
	float kr = 0.5f * (r_parallel * r_parallel + r_perpendicular * r_perpendicular);

	return kr;
}
void FresnelReflector::setEtaIn(float etain) {
	mEtaIn = etain;
}
void FresnelReflector::setEtaOut(float etaout) {
	mEtaOut = etaout;
}
void FresnelReflector::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(100);
}
// ***** Lambertian function members *****
Lambertian::Lambertian() : mDiffuseColour{}, mDiffuseReflection{}, mSampler{}
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
Lambertian::sample_f([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& incoming, float& pdf) const
{
	using atlas::math::Vector;

	Vector w = sr.normal;
	Vector v = glm::normalize(glm::cross(Vector(0.0034, 1.0, 0.0071), w));
	Vector u = glm::cross(v, w);

	atlas::math::Point sp = mSampler->sampleHemiSphere();
	Vector wi = sp.x * u + sp.y * v + sp.z * w;
	pdf = glm::dot(sr.normal, wi) * glm::one_over_pi<float>();

	return mDiffuseColour * mDiffuseReflection * glm::one_over_pi<float>();;
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
void Lambertian::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(1);
}
void Lambertian::setDiffuseColour(Colour const& colour)
{
	mDiffuseColour = colour;
}
PerfectSpecular::PerfectSpecular() : mPerfectColour{}, kr{}
{}
PerfectSpecular::PerfectSpecular(Colour pPerfectColour, float pkr) :
	mPerfectColour{ pPerfectColour },
	kr{ pkr }
{}
Colour PerfectSpecular::fn([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector const& incoming) const {

	return { 0.0, 0.0, 0.0 };
}
Colour PerfectSpecular::sample_f(ShadeRec const& sr,
	atlas::math::Vector const& reflected,
	atlas::math::Vector& incoming) const {

	float ndotwo = glm::dot(sr.normal, reflected);
	incoming = -reflected + 2.0f * sr.normal * ndotwo;
	return (kr * mPerfectColour / (glm::dot(sr.normal, incoming)));
}
Colour PerfectSpecular::sample_f(ShadeRec const& sr,
	atlas::math::Vector const& reflected,
	atlas::math::Vector& incoming, float& pdf) const {

	float ndotwo = glm::dot(sr.normal, reflected);
	incoming = -reflected + 2.0f * sr.normal * ndotwo;
	pdf = glm::dot(sr.normal, incoming);

	return (kr * mPerfectColour / (glm::dot(sr.normal, incoming)));
}
Colour PerfectSpecular::rho([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected) const {

	return { 0.0, 0.0, 0.0 };
}
void PerfectSpecular::setKr(float pkr) {
	kr = pkr;
}
void PerfectSpecular::setPerfectColour(Colour const& colour) {
	mPerfectColour = colour;
}
void PerfectSpecular::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(1);
}

// ******* GlossySpecular function members *******
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


void GlossySpecular::setKr(float pkr)
{
	kr = pkr;
}
void GlossySpecular::setExponent(float ep)
{
	mExponent = ep;
}

void GlossySpecular::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(mExponent);
}
Colour GlossySpecular::sample_f([[maybe_unused]] ShadeRec const& sr,
	[[maybe_unused]] atlas::math::Vector const& reflected,
	[[maybe_unused]] atlas::math::Vector& incoming, [[maybe_unused]] float& pdf) const
{
	float ndotwo = glm::dot(sr.normal, reflected);
	atlas::math::Vector r = -reflected + 2.0f * sr.normal * ndotwo;

	atlas::math::Vector w = r;
	atlas::math::Vector u = glm::normalize(glm::cross(atlas::math::Vector(0.00424, 1, 0.00764), w));
	atlas::math::Vector v = glm::cross(u, w);

	atlas::math::Point sp = mSampler->sampleHemiSphere();
	incoming = sp.x * u + sp.y * v + sp.z * w;

	if (glm::dot(sr.normal, incoming) < 0.0f)
	{
		incoming = -sp.x * u - sp.y * v + sp.z * w;
	}
	float phong_lobe = pow(glm::dot(r, incoming), mExponent);
	pdf = phong_lobe * glm::dot(sr.normal, incoming);

	return (kr * mSpecularColour * phong_lobe);
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

// ***** Dielectric function members *****
Dielectric::Dielectric() : mFresnelBRDF{}, mFresnelBTDF{}
{
	mFresnelBRDF = std::make_shared<FresnelReflector>();
	mFresnelBTDF = std::make_shared<FresnelTransmitter>();
}

Dielectric::Dielectric(Colour mColourIn, Colour mColourOut) : Dielectric()
{
	setColourIn(mColourIn);
	setColourOut(mColourOut);
}
void Dielectric::setColourIn(Colour colorin) {
	mColourIn = colorin;
	mFresnelBTDF->setColour(colorin);
	setSpecularColour(colorin);
}
void Dielectric::setColourOut(Colour colorout) {
	mColourOut = colorout;
	mFresnelBRDF->setColour(colorout);
	setSpecularColour(colorout);
}
void Dielectric::setEtaIn(float etain) {
	mFresnelBRDF->setEtaIn(etain);
	mFresnelBTDF->setEtaIn(etain);
}
void Dielectric::setEtaOut(float etaout) {
	mFresnelBRDF->setEtaOut(etaout);
	mFresnelBTDF->setEtaOut(etaout);
}
Colour Dielectric::shade(ShadeRec& sr)
{
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };

	Vector wi;
	Vector wo = -sr.ray.d;
	Colour fr = mFresnelBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);
	float t;
	Colour Lr, Lt;
	float ndotwi = glm::dot(sr.normal, wi);

	if (mFresnelBTDF->tir(sr))
	{
		if (ndotwi < 0.0f)
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;
		}
		else
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;
		}
	}
	else
	{
		Vector wt;
		Colour ft = mFresnelBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);
		float ndotwt = glm::dot(sr.normal, wt);

		if (ndotwi < 0.0f)
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * std::fabs(ndotwi);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;

			Lt = ft * sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) * std::fabs(ndotwt);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lt;
		}
		else
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * fabs(ndotwi);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;

			Lt = ft * sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) * fabs(ndotwt);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lt;
		}
	}
	return L;
}
Colour Dielectric::pathShade([[maybe_unused]] ShadeRec& sr)
{
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };

	Vector wi;
	Vector wo = -sr.ray.d;
	Colour fr = mFresnelBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);
	float t = 0.0f;
	Colour Lr, Lt;
	float ndotwi = glm::dot(sr.normal, wi);

	if (mFresnelBTDF->tir(sr))
	{
		if (ndotwi < 0.0f)
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;
		}
		else
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;
		}
	}
	else
	{
		Vector wt;
		Colour ft = mFresnelBTDF->sample_f(sr, wo, wt);
		//printf("\nfr %f %f %f\n", fr.x, fr.y, fr.z);
		Ray transmitted_ray(sr.hit_point, wt);
		float ndotwt = glm::dot(sr.normal, wt);

		if (ndotwi < 0.0f)
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * std::fabs(ndotwi);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;

			Lt = ft * sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) * std::fabs(ndotwt);
			//printf("Lt %f %f %f\n", Lt.x, Lt.y, Lt.z);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lt;
		}
		else
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * fabs(ndotwi);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;

			Colour color{ sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) };
			//printf("color %f %f %f\n", color.x, color.y, color.z);
			Lt = ft * color * fabs(ndotwt);
			//printf("fabs %f\n", fabs(ndotwt));
			//printf("Lt2 %f %f %f\n", Lt.x, Lt.y, Lt.z);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lt;
		}
	}
	return L;
}
Colour Dielectric::areaLightShade(ShadeRec& sr) {
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };

	Vector wi;
	Vector wo = -sr.ray.d;
	Colour fr = mFresnelBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);
	float t;
	Colour Lr, Lt;
	float ndotwi = glm::dot(sr.normal, wi);

	if (mFresnelBTDF->tir(sr))
	{
		if (ndotwi < 0.0f)
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;
		}
		else
		{
			Lr = sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;
		}
	}
	else
	{
		Vector wt;
		Colour ft = mFresnelBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);
		float ndotwt = glm::dot(sr.normal, wt);

		if (ndotwi < 0.0f)
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * std::fabs(ndotwi);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lr;

			Lt = ft * sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) * std::fabs(ndotwt);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lt;
		}
		else
		{
			Lr = fr * sr.world->tracer->trace_ray(reflected_ray, t, sr.depth + 1) * fabs(ndotwi);
			L += Colour{ powf(mColourOut.x, t), powf(mColourOut.y, t), powf(mColourOut.z, t) } *Lr;

			Lt = ft * sr.world->tracer->trace_ray(transmitted_ray, t, sr.depth + 1) * fabs(ndotwt);
			L += Colour{ powf(mColourIn.x, t), powf(mColourIn.y, t), powf(mColourIn.z, t) } *Lt;
		}
	}
	return L;

}
Colour Dielectric::L([[maybe_unused]] ShadeRec& sr) {
	return Colour{ 0, 0, 0 };
}
void Dielectric::setSampler(std::shared_ptr<Sampler> sampler) {
	mFresnelBRDF->setSampler(sampler);
	mFresnelBTDF->setSampler(sampler);
}


// ***** Transparent function members *****
Transparent::Transparent() : mReflectiveBRDF{}, mSpecularBTDF{}
{
	mReflectiveBRDF = std::make_shared<PerfectSpecular>();
	mSpecularBTDF = std::make_shared<PerfectTransmitter>();
}

Transparent::Transparent(float ior, float kr, float kt) : Transparent()
{
	setIor(ior);
	setKr(kr);
	setKt(kt);
}
void Transparent::setColour(Colour color) {
	mSpecularBTDF->setColour(color);
	mReflectiveBRDF->setPerfectColour(color);
	setSpecularColour(color);
}
Colour Transparent::shade(ShadeRec& sr)
{
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };

	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);

	if (mSpecularBTDF->tir(sr)) {
		L += sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1);
	}
	else {
		Vector wt;
		Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);

		L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wi));

		L += fr * sr.world->tracer->trace_ray(transmitted_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wt));
	}
	return L;
}
Colour Transparent::pathShade([[maybe_unused]] ShadeRec& sr)
{
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };

	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);

	if (mSpecularBTDF->tir(sr)) {
		L += sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1);
	}
	else {
		Vector wt;
		Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);

		L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wi));

		L += ft * sr.world->tracer->trace_ray(transmitted_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wt));
	}
	return L;
}
Colour Transparent::areaLightShade(ShadeRec& sr) {
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };

	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mReflectiveBRDF->sample_f(sr, wo, wi);
	Ray reflected_ray(sr.hit_point, wi);

	if (mSpecularBTDF->tir(sr)) {
		L += sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1);
	}
	else {
		Vector wt;
		Colour ft = mSpecularBTDF->sample_f(sr, wo, wt);
		Ray transmitted_ray(sr.hit_point, wt);

		L += fr * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wi));

		L += fr * sr.world->tracer->trace_ray(transmitted_ray, sr.depth + 1) *
			std::fabs(glm::dot(sr.normal, wt));
	}
	return L;

}
Colour Transparent::L([[maybe_unused]] ShadeRec& sr) {
	return Colour{ 0, 0, 0 };
}
void Transparent::setSampler(std::shared_ptr<Sampler> sampler) {
	mReflectiveBRDF->setSampler(sampler);
	mSpecularBTDF->setSampler(sampler);
}
void Transparent::setIor(float ior)
{
	mSpecularBTDF->setIor(ior);
}

void Transparent::setKr(float kr)
{
	mReflectiveBRDF->setKr(kr);
}

void Transparent::setKt(float kt)
{
	mSpecularBTDF->setKt(kt);
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
Colour Emissive::pathShade([[maybe_unused]] ShadeRec& sr)
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

// ***** GlossyReflector function members *****
GlossyReflector::GlossyReflector() :
	Phong{},
	mGlossyBRDF{ std::make_shared<GlossySpecular>() }
{}
GlossyReflector::GlossyReflector(float kr, Colour color) : GlossyReflector{}
{
	mGlossyBRDF->setKr(kr);
	mGlossyBRDF->setSpecularColour(color);
	setKr(kr);
	setSpecularColour(color);
}
void GlossyReflector::setKs(float ks) {
	setSpecularReflection(ks);
	mGlossyBRDF->setSpecularReflection(ks);
}
void GlossyReflector::setKa(float ka)
{
	setAmbientReflection(ka);
}
void GlossyReflector::setKd(float kd) {
	setSpecularReflection(kd);
	mGlossyBRDF->setDiffuseReflection(kd);
}
void GlossyReflector::setExponent(float ex) {
	Phong::setExponent(ex);
	mGlossyBRDF->setExponent((ex));
}

void GlossyReflector::setKr(float kr)
{
	mGlossyBRDF->setKr(kr);
}
void GlossyReflector::setSampler(std::shared_ptr<Sampler> sampler)
{
	mGlossyBRDF->setExponent(getExponent());
	mGlossyBRDF->setSampler(sampler);
}
Colour GlossyReflector::L([[maybe_unused]] ShadeRec& sr)
{
	return { 0, 0, 0 };
}
Colour GlossyReflector::shade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;

	Colour fr = mGlossyBRDF->sample_f(sr, wo, wi);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi);

	return L;
}
Colour GlossyReflector::pathShade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;
	float pdf;
	Colour fr = mGlossyBRDF->sample_f(sr, wo, wi, pdf);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi) / pdf;

	return L;
}
Colour GlossyReflector::areaLightShade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;
	float pdf;
	Colour fr = mGlossyBRDF->sample_f(sr, wo, wi, pdf);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi) / pdf;

	return L;
}
// ***** Reflective function members *****
Reflective::Reflective() :
	Phong{},
	mPerfectBRDF{ std::make_shared<PerfectSpecular>() }
{}
Reflective::Reflective(float kr, Colour color) : Reflective{}
{
	mPerfectBRDF->setKr(kr);
	mPerfectBRDF->setPerfectColour(color);
	setSpecularColour(color);
}
void Reflective::setKr(float kr) {
	mPerfectBRDF->setKr(kr);
	Phong::setKr(kr);
}
Colour Reflective::L([[maybe_unused]] ShadeRec& sr)
{
	return { 0, 0, 0 };
}
Colour Reflective::shade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::shade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mPerfectBRDF->sample_f(sr, wo, wi);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi);

	return L;
}
Colour Reflective::pathShade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;
	float pdf;
	Colour fr = mPerfectBRDF->sample_f(sr, wo, wi, pdf);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi) / pdf;
	return L;
}
Colour Reflective::areaLightShade(ShadeRec& sr) {
	using atlas::math::Ray;
	using atlas::math::Vector;

	Colour L{ Phong::areaLightShade(sr) };
	Vector wo = -sr.ray.d;
	Vector wi;
	Colour fr = mPerfectBRDF->sample_f(sr, wo, wi);
	Ray shadowRay(sr.hit_point, wi);

	L += fr * sr.world->tracer->trace_ray(shadowRay, sr.depth + 1) * glm::dot(sr.normal, wi);

	return L;
}
void Reflective::setSampler(std::shared_ptr<Sampler> sampler) {
	Phong::setSampler(sampler);
	mPerfectBRDF->setSampler(sampler);
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
void Phong::setKr(float kr)
{
	mSpecularBRDF->setKr(kr);
}
void Phong::setSpecularColour(Colour colour)
{
	mDiffuseBRDF->setDiffuseColour(colour);
	mAmbientBRDF->setDiffuseColour(colour);
	mSpecularBRDF->setSpecularColour(colour);
}
float Phong::getExponent() {
	return mExponent;
}
void Phong::setSampler(std::shared_ptr<Sampler> sampler) {
	mAmbientBRDF->setSampler(sampler);
	mDiffuseBRDF->setSampler(sampler);
	mSpecularBRDF->setSampler(sampler);
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
	//printf("mAmbient %f %f %f\n", L.x, L.y, L.z);
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
Colour Phong::pathShade([[maybe_unused]] ShadeRec& sr)
{
	using atlas::math::Vector;

	Vector wi;
	Vector wo = -sr.ray.d;
	float pdf;
	Colour f = mSpecularBRDF->sample_f(sr, wo, wi, pdf);
	float ndotwi = glm::dot(sr.normal, wi);
	Ray reflected_ray(sr.hit_point, wi);

	return (f * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) * ndotwi / pdf);
}
Colour Phong::areaLightShade(ShadeRec& sr)
{
	using atlas::math::Ray;
	using atlas::math::Vector;

	Vector wo = -sr.ray.d;
	Colour L = mAmbientBRDF->rho(sr, wo) * sr.world->ambient->L(sr);

	std::size_t numLights = sr.world->lights.size();

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

				L += (mDiffuseBRDF->fn(sr, wo, wi) + mSpecularBRDF->fn(sr, wo, wi)) *
					sr.world->lights[i]->L(sr)
					* nDotWi * sr.world->lights[i]->G(sr) / sr.world->lights[i]->pdf(sr);
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
void Matte::setSampler(std::shared_ptr<Sampler> sampler) {
	mAmbientBRDF->setSampler(sampler);
	mDiffuseBRDF->setSampler(sampler);
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
Colour Matte::pathShade(ShadeRec& sr)
{
	using atlas::math::Vector;
	Vector wi;
	Vector wo = -sr.ray.d;
	float pdf;
	Colour f = mDiffuseBRDF->sample_f(sr, wo, wi, pdf);
	float ndotwi = glm::dot(sr.normal, wi);
	Ray reflected_ray(sr.hit_point, wi);

	return (f * sr.world->tracer->trace_ray(reflected_ray, sr.depth + 1) * ndotwi / pdf);
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

// ***** AmbientOccluder function members *****
AmbientOccluder::AmbientOccluder() :Light{}, mU{ atlas::math::Vector{0, 0, 0} },
mV{ atlas::math::Vector{0, 0, 0} }, mW{ atlas::math::Vector{0, 0, 0} }, minAmount{ 0.0 }
{}
void AmbientOccluder::setMinAmount(float pMinMount) {
	minAmount = pMinMount;
}
void AmbientOccluder::setSampler(std::shared_ptr<Sampler> sampler)
{
	mSampler = sampler;
	mSampler->mapSamplesToHemisphere(1.0);
}
atlas::math::Vector AmbientOccluder::getDirection([[maybe_unused]] ShadeRec& sr)
{
	atlas::math::Point sp = mSampler->sampleHemiSphere();
	return (sp.x * mU + sp.y * mV + sp.z * mW);
}
bool AmbientOccluder::in_shadow(atlas::math::Ray<atlas::math::Vector> const& ray, ShadeRec& sr) const
{
	float t;
	std::size_t numObjects = sr.world->objects.size();

	for (std::size_t j{ 0 }; j < numObjects; j++) {
		if (sr.world->objects[j]->intersectRay(ray, t)) {
			return true;
		}
	}
	return false;
}
Colour AmbientOccluder::L(ShadeRec& sr)
{
	using atlas::math::Ray;

	mW = sr.normal;
	mV = glm::normalize(glm::cross(mW, atlas::math::Vector(0.0072, 1.0, 0.0034)));
	mU = glm::cross(mV, mW);

	Ray shadowRay(sr.hit_point, getDirection(sr));

	if (in_shadow(shadowRay, sr)) {
		return (minAmount * mRadiance * mColour);
	}
	else {
		return mRadiance * mColour;
	}
}
float AmbientOccluder::G([[maybe_unused]] ShadeRec& sr)
{
	return 1.0;
}
float AmbientOccluder::pdf([[maybe_unused]] ShadeRec& sr)
{
	return 1.0;
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
	float t = std::numeric_limits<float>::max();
	std::size_t num_objects = sr.world->objects.size();

	float ts = glm::dot(atlas::math::Vector(samplePoint - ray.o), ray.d);
	for (int i = 0; i < num_objects; ++i)
	{
		if (sr.world->objects[i]->intersectRay(ray, t) && t < ts) {
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
Colour AreaLight::L(ShadeRec& sr)
{
	float ndotd = glm::dot(-normal, wi);
	if (ndotd > 0.0f) {
		return mMaterial->L(sr);
	}
	else {
		return { 0, 0, 0 };
	}
}
float AreaLight::G(ShadeRec& sr)
{
	float ndotd = glm::dot(-normal, wi);

	float d2 = (samplePoint.x - sr.hit_point.x) * (samplePoint.x - sr.hit_point.x)
		+ (samplePoint.y - sr.hit_point.y) * (samplePoint.y - sr.hit_point.y)
		+ (samplePoint.z - sr.hit_point.z) * (samplePoint.z - sr.hit_point.z);

	return (ndotd / d2);
}
float AreaLight::pdf(ShadeRec& sr) {
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
	std::size_t num_objects = sr.world->objects.size();
	float d = glm::length(mLocation - ray.o);

	for (int i = 0; i < num_objects; ++i) {
		if (sr.world->objects[i]->intersectRay(ray, t) && t < d) {
			return true;
		}
	}

	return false;
}
Colour PointLight::L([[maybe_unused]] ShadeRec& sr)
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
	std::size_t num_objects = sr.world->objects.size();

	for (int i = 0; i < num_objects; ++i) {
		if (sr.world->objects[i]->intersectRay(ray, t) && atlas::core::geq(t, kEpsilon)) {
			return true;
		}
	}
	return false;
}
Colour Directional::L([[maybe_unused]] ShadeRec& sr)
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
Colour Ambient::L([[maybe_unused]] ShadeRec& sr)
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
	std::shared_ptr<World> world{ std::make_shared<World>(1000, 1000, Colour{0, 0, 0}, 3) };

	world->setSampler(std::make_shared<Jittered>(9, 80));

	world->build();

	//pinhole camera
	std::shared_ptr<Pinhole> pinhole{ std::make_shared<Pinhole>() };
	pinhole->setEye({ 0.0f, -200.0f, 300.0f });
	pinhole->computeUVW();
	pinhole->setWorld(world);
	world->setCamera(pinhole);

	//tracer
	std::shared_ptr<PathTrace> tracer{ std::make_shared<PathTrace>(world) };
	world->setTracer(tracer);

	pinhole->renderScene();

	saveToFile("render.bmp", world->width, world->height, world->image);

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