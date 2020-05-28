import numpy as np

class AssemblyClassifier:
    def __init__(self, input_dim, n_assembly=10000, n_cap=100, edge_prob=0.01, beta = 0.1, initial_projection=True):
        self._d = input_dim
        self._initial_projection = initial_projection
        if initial_projection:
            self._n = n_assembly
        else:
            self._n = input_dim
        self._k = n_cap
        self._p = edge_prob
        self._B = beta
        if initial_projection:
            self.W_rp = np.random.binomial(1,self._p,size=(self._n,self._d)).astype("float64")
        self.W_oi = np.random.binomial(1,self._p,size=(2*self._n,self._n)).astype("float64")
        self.W_oo = np.random.binomial(1,self._p,size=(2*self._n,2*self._n)).astype("float64")

    def get_output_activities(self, X, num_timesteps=1):
        num_inputs = X.shape[0]
        activities = np.zeros((num_inputs,2))

        if self._initial_projection:
            inputs = X.dot(self.W_rp.T)
            inputs = np.array([self.cap(inputs[i]) for i in range(num_inputs)])
        else:
            inputs = X
        for i in range(num_inputs):
            y_tm1 = np.zeros(2*self._n)
            for t in range(num_timesteps):
                y_t = self.W_oi.dot(inputs[i]) + self.W_oo.dot(y_tm1)
                y_t = self.cap(y_t)
                y_tm1 = np.copy(y_t)
            activities[i] = np.array([sum(y_t[:self._n//2]), sum(y_t[self._n//2:])])
        return activities

    def predict(self, X, num_timesteps=1):
        activities = self.get_output_activities(X, num_timesteps=num_timesteps)
        outputs = np.argmax(activities, axis=1)
        return outputs

    def accuracy(self, X, y, num_timesteps=1):
        preds = self.predict(X, num_timesteps=num_timesteps)
        return sum(preds == y)

    def cap(self, arr):
        """
        perform the cap operation in any assembly
        """
        if len(np.where(arr !=0)[0]) > self._k:
            indices = np.argsort(arr)
            arr[indices[:-self._k]]=0
        arr[np.where(arr != 0.)[0]] = 1.0
        return arr


    def train_cap(self, arr, label):
        """
        perform the cap operation in the output assembly:
        first half of neurons in the output assembly correspond to zero
        and the second half correspond to one
        """
        if len(np.where(arr !=0)[0]) > self._k:
            indices = np.argsort(arr)
            arr[indices[:-self._k]]=0

        arr[np.where(arr != 0.)[0]] = 1.0
        if label==0:
            arr[self._n//2:] = 0.
        else:
            arr[:self._n//2] = 0.
        return arr

    def train(self, X, y, num_timesteps=3):
        num_inputs = X.shape[0]
        EPS = 1e-20

        if self._initial_projection:
            inputs = X.dot(self.W_rp.T)
            inputs = np.array([self.cap(inputs[i]) for i in range(num_inputs)])
        else:
            inputs = X

        for t in range(num_inputs):
            y_tm1 = np.zeros(2*self._n)

            # i steps of firing impulses
            for _ in range(num_timesteps):
                y_t = self.W_oi.dot(inputs[t]) + self.W_oo.dot(y_tm1)
                y_t = self.train_cap(y_t, y[t])
                y_tm1 = np.copy(y_t)

            # plasticity modifications
            for i in np.where(y_t!=0)[0]:
                for j in np.where(inputs[t]!=0)[0]:
                    self.W_oi[i,j] *= 1.+self._B

            for i in np.where(y_t!=0)[0]:
                for j in np.where(y_tm1!=0)[0]:
                    self.W_oo[i,j] *= 1.+self._B

            if (t+1)%10 == 0:
                self.W_oi = np.diag(1./(EPS+self.W_oi.dot(np.ones(self.W_oi.shape[1])))).dot(self.W_oi)
                self.W_oo = np.diag(1./(EPS+self.W_oo.dot(np.ones(self.W_oo.shape[1])))).dot(self.W_oo)
