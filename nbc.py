from scipy.spatial import cKDTree
import numpy as np


class NBC:

    def __init__(self, k, dataset) -> None:
        self.k = k
        self.index = self.build_KDtree(dataset)
        self.dataset = dataset
        self.epsilon = 10e-8

    def build_KDtree(self, dataset):

        kdtree = cKDTree(dataset)

        return kdtree

    def _k_nearest_neighbors_max_radius(self, query_point, point_id):
        
        # Perform a k-nearest neighbor query and convert the generator to a list
        _ , neighbors_with_point = list(self.index.query(query_point, self.k + 1))  # k + 1 to include the point itself
        
        # Manually exclude the point itself from the list
        mask = neighbors_with_point != point_id
        
        distances = np.array([self.euclidean_distance(query_point, self.dataset[neighbor]) for neighbor in neighbors_with_point])

        distances = distances[mask]
        
        return max(distances)


    def _find_neighbors_circle(self, point, point_id, neighborhood_radius):
        
        # Perform a range query to find neighbors within the circular radius
        neighbors_with_point = self.index.query_ball_point(point, neighborhood_radius+self.epsilon)
        neighbors_with_point = np.array([neighbor for neighbor in neighbors_with_point if self.euclidean_distance(point, self.dataset[neighbor] <= neighborhood_radius)])

        neighbors = neighbors_with_point[neighbors_with_point!=point_id]

        return neighbors

    def R_kNN(self, point, point_id):

        max_radius = self._k_nearest_neighbors_max_radius(point, point_id)

        return self._find_neighbors_circle(point, point_id, max_radius)

    
    def get_neigbourhoods(self):

        neighbours = [[] for _ in range(len(self.dataset))]
        whos_neighbour_count = np.zeros(len(self.dataset), dtype=int)

        for i, point in enumerate(self.dataset):
            point_neighbours = self.R_kNN(point, i)
            neighbours[i] = point_neighbours
            
            whos_neighbour_count[point_neighbours] += 1

        return neighbours, whos_neighbour_count
    
    def euclidean_distance(self, point1, point2):

        return np.linalg.norm(point1 - point2)
    
    def get_ndf(self, neighbours, whos_neighbour_count):
        ndf = np.zeros(len(self.dataset))

        for i in range(len(self.dataset)):
            ndf[i] = whos_neighbour_count[i] / len(neighbours[i])

        return ndf

    # def get_clusters(self, ndf, neighbours):

        # clusters = [None for _ in range(len(self.dataset))]
        
        # cld=0
        # seeds = []

        # for i in range(len(self.dataset)):
        #     if clusters[i] != None:
        #         continue

        #     if ndf[i] < 1:
        #         clusters[i] = 'N'
        #     else:
        #         clusters[i] = cld
        #         seeds.append(i)
             
        #         while seeds:
        #             point = seeds.pop(0)
        #             for neighbour in neighbours[point]:

        #                 if clusters[neighbour] == 'N':
        #                     clusters[neighbour] = cld
        #                 elif clusters[neighbour] is None:
        #                     clusters[neighbour] = cld
        #                     if ndf[neighbour] >= 1:
        #                         seeds.append(neighbour)
        #         cld += 1



        # return clusters
    
    def get_clusters(self, ndf, neighbours):

        clusters = ['N' for _ in range(len(self.dataset))]
        
        cld=0

        for i in range(len(self.dataset)):
            if clusters[i] != 'N' or ndf[i] < 1: continue

            clusters[i] = cld
            dpset = []
             
            for neighbour in neighbours[i]:
                clusters[neighbour] = cld
                if ndf[neighbour] >= 1:
                    dpset.append(neighbour)

            while dpset:
                point = dpset.pop(0)
                for neighbour in neighbours[point]:

                    if clusters[neighbour] != 'N': continue

                    clusters[neighbour] = cld
                    if ndf[neighbour] >= 1:
                        dpset.append(neighbour)

            cld += 1

        return clusters


    def run(self):

        neighbours, whos_neighbour_count = self.get_neigbourhoods()
        
        ndf = self.get_ndf(neighbours, whos_neighbour_count)

        clusters = self.get_clusters(ndf, neighbours)

        return clusters


    

if __name__ == '__main__':
    dataset = np.array([
        np.array([4.2, 4.0]),
        np.array([5.9, 3.9]),
        np.array([2.8, 3.5]),
        np.array([12.0, 1.3]),
        np.array([10.0, 1.3]),
        np.array([1.1, 3.0]),
        np.array([0.0, 2.4]),
        np.array([2.4, 2.0]),
        np.array([11.5, 1.8]),
        np.array([11.0, 1.0]),
        np.array([0.9, 0.0]),
        np.array([1.0, 1.5]),
    ]) # Your dataset
    k = 3
    nbc = NBC(k, dataset)

    print(nbc.run())