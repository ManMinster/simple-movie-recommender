from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import combinations
from math import sqrt
import os


class MovieRecommender(MRJob):

	def configure_options(self):
		super(MovieRecommender, self).configure_options()
		self.add_file_option('--items', help = 'file path to u.item')

	def load_movie_names(self):
		self.movieNames = {}

		with open(os.getcwd().join("/data/u.item")) as f:
			for line in f:
				fields = line.split("|")
				self.movieNames[int(fields[0])] = fields[1]

	def load_items(self):
		self.genres ={}

		with open(os.getcwd().join("/data/u.item")) as f:
			for line in f:
				fields = line.split("|")
				self.genres[int(fields[0])] = int(''.join(fields[5:23]))

	def steps(self):
		return [
			MRStep(mapper=self.mapper_parse_file,
				reducer=self.reducer_all_ratings_per_user),
			MRStep(mapper=self.mapper_create_movie_pairs,
				reducer=self.reducer_compute_similarity),
			MRStep(mapper_init=self.load_movie_names,
				mapper=self.mapper_sort_similarities,
				reducer=self.reducer_output_similarities)
			]

	def mapper_parse_file(self, key, line):
		(userID, movieID, rating, timestamp) = line.split("\t")
		yield userID, (movieID, float(rating))

	def reducer_all_ratings_per_user(self, userID, itemRating):
		ratings = []

		for movieID, rating in itemRating:
			ratings.append((userID, rating))

		yield userID, ratings

	def mapper_create_movie_pairs(self, userID, itemRatings):
		for itemRating1, itemRating2 in combinations(itemRatings, 2):
			movieID1 = itemRating1[0]
			rating1	 = itemRating1[1]
			movieID2 = itemRating2[0]
			rating2  = itemRating2[1]

		yield (movieID1, movieID2), (rating1, rating2)
		yield (movieID2, movieID1), (rating2, rating1)


	def cosine_similarity(self, ratingPairs):
		numPairs = 0
		sum_xx = sum_yy = sum_xy = 0

		# calculate cosine similarity (angle) between vector x and y, ie, between the rating pair
		#    VECTOR X     VECTOR Y
		for componentX, componentY in ratingPairs:
			sum_xx += componentX * componentX
			sum_yy += componentY * componentY
			sum_xy += componentX * componentY
			numPairs += 1

		dotProduct = sum_xy
		lengthVectorX = sqrt(sum_xx) # just like finding the length of hyp in pythagoras triangle
		lengthVectorY = sqrt(sum_yy)
		denominator = lengthVectorX * lengthVectorY

		cosSimilarity = 0

		if (denominator):
			cosSimilarity = (dotProduct) / (float(denominator))

		return (cosSimilarity, numPairs)

	def reducer_compute_similarity(self, moviePair, ratingPairs):
		cosSimilarity, numPairs = self.cosine_similarity(ratingPairs)
		if (numPairs > 1 and cosSimilarity > 0.7):
			yield moviePair, (cosSimilarity, numPairs)

	def mapper_sort_similarities(self, moviePair, scores):
		cosSimilarity, numPairs = scores
		movie1, movie2 = moviePair

		yield (self.movieNames[int(movie1)], cosSimilarity), \
				(self.movieNames[float(movie2)], numPairs)


	def reducer_output_similarities(self, movieNameAndScore, similarityN):
		movie1, score = movieNameAndScore

		# movie => Similar movie, similarity score, number of co-ratings
		for movie2, numPairs in similarityN:
			yield movie1, (movie2, score, numPairs)

if __name__ == "__main__":
	MovieRecommender.run()
