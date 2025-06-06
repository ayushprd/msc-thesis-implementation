import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jaxopt import GradientDescent, LBFGS
import numpy as np
from scipy.spatial import distance


# Logit
def logit(p):
    return jnp.log(p / (1.0 - p))


def caverage(y, pseudocount=1.0):
    return logit(
        (jnp.sum(0.5 + 0.5 * y) + 0.5 * pseudocount) / (y.shape[0] + pseudocount)
    )


def knnpred(Ztest, Z, y, k=5, average=jnp.average):
    # Compute all L1 distances from test points to the training data
    distances = jnp.sum(jnp.abs(Ztest[:, None, :] - Z[None, :, :]), axis=2)
    # Find indices of k nearest neighbours
    kni = jnp.argsort(distances, axis=1)[:, :k]
    # return averaged target values
    return vmap(lambda i: average(y[i]))(kni)


def softplus(x, beta=1):
    # calculate (1/beta) * np.log(1 + np.exp(beta * x))
    return (1 / beta) * jnp.logaddexp(beta * x, 0)


# Initialize parameters for one weak learner
def init_network_params1(p, b, key):
    a_key, w_key = random.split(key, 2)
    # First layer weights and biases
    w = random.normal(w_key, (p,))
    a = random.normal(a_key)
    b = jnp.array(b)
    return (a, b, w)


# Forward pass of one weak learner
def predict1(params, x, scale=1.0):
    a, b, w = params
    # return a + b * softplus(scale * jnp.dot(x, w.T)) / scale
    return a + b * softplus(x=jnp.dot(x, w.T), beta=scale)


# Quadratic loss
def loss_quadratic(y, yp):
    return jnp.average((y - yp) ** 2)


# Logistic loss
def loss_logistic(y, yp):
    return jnp.average(softplus(-y * yp))


def add_intercept(x):
    if len(x.shape) == 1:
        x = x[None, :]
    return jnp.hstack((x, jnp.ones((x.shape[0], 1))))


# One boosting iteration: learn weak learner
def learn1(
    x,
    y,
    y0,
    params0,
    optimizer_params,
    loss=loss_quadratic,
    ridge=1e-3,
    softplus_scale=1,
    optimizer=LBFGS,
):
    a, b, w = params0
    par0 = a, w

    def objective_fn(par):
        a, w = par
        params = a, b, w
        return loss(
            y, y0 + predict1(params=params, x=x, scale=softplus_scale)
        ) + ridge * jnp.average(w**2)

    # lbfgs = LBFGS(fun=objective_fn, maxiter=100)
    solver = optimizer(fun=objective_fn, **optimizer_params)
    opt_result = solver.run(init_params=par0)
    # Extract optimized parameters
    a, w = opt_result.params
    return (a, b, w)


# m boosting iterations
def learn(
    m,
    x,
    y,
    key,
    optimizer_params,
    ridge=1e-3,
    y0=None,
    loss=loss_quadratic,
    softplus_scale=1,
    randomise=True,
    optimizer=LBFGS,
):
    if y0 is None:
        y0 = jnp.zeros(y.shape)

    keys = random.split(key, m)

    def learnf(y0, b, key):
        params0 = init_network_params1(x.shape[1], b, key)

        params = learn1(
            x=x,
            y=y,
            y0=y0,
            params0=params0,
            optimizer_params=optimizer_params,
            loss=loss,
            ridge=ridge,
            softplus_scale=softplus_scale,
            optimizer=optimizer,
        )
        predict = predict1(params=params, x=x, scale=softplus_scale)
        lossv = loss(y, y0 + predict)
        return params, lossv

    learnf_jit = jit(learnf)

    params = [None] * m
    losses = [None] * m

    for j in range(m):
        params_plus, loss_plus = learnf_jit(y0, 1.0, keys[j])
        params_minus, loss_minus = learnf_jit(y0, -1.0, keys[j])
        params[j] = params_minus if loss_minus < loss_plus else params_plus
        losses[j] = (loss_minus if loss_minus < loss_plus else loss_plus).item()
        y0 = y0 + predict1(params[j], x, scale=softplus_scale)

    return params, jnp.array(losses)


def learn_beta(x, y, y0, params, loss=loss_quadratic, ridge=1e-3):
    if y0 is None:
        y0 = jnp.zeros(y.shape)

    beta0 = jnp.ones(len(params))
    z = np.array([predict1(parami, x) for parami in params]).T

    def objective_fn(beta):
        z_weighted = z * beta
        yhat = np.sum(z_weighted, axis=1)
        return loss(y, y0 + yhat) + ridge * jnp.average(beta**2)

    optimizer = LBFGS(fun=objective_fn, maxiter=100)
    opt_result = optimizer.run(init_params=beta0)
    # Extract optimized parameters
    beta = opt_result.params
    return beta


def predict(params, x, y0=0.0, softplus_scale=1):
    z = jnp.array([predict1(parami, x, softplus_scale) for parami in params]).T
    return y0 + jnp.sum(z, axis=1)


# Logit
def logit(p):
    return jnp.log(p / (1.0 - p))


def score_to_labels(score):
    return np.where(score > 0, 1, -1)


# Split dataset into two roughly equally sized halves by a given feature (a and b).
# Then split these halves into two in random (1 and 2), resulting to four blocks in total (a1, a2, b1, b2).
def split4(key, X, feature=0):
    key_a, key_b = random.split(key)
    i = jnp.argsort(X[:, feature])
    ia = i[: (X.shape[0] // 2)]
    ib = i[(X.shape[0] // 2) :]
    ia1 = random.choice(key_a, ia, (ia.shape[0] // 2,), replace=False)
    ia2 = jnp.setdiff1d(ia, ia1)
    ib1 = random.choice(key_b, ib, (ib.shape[0] // 2,), replace=False)
    ib2 = jnp.setdiff1d(ib, ib1)
    return ia1, ia2, ib1, ib2


class GBMAP4:
    """GradientBoostMAP class"""

    def __init__(
        self,
        n_boosts=10,  # boosting iterations
        optim_maxiter=100,  # max iterations for gradient descent (for finding alpha)
        optim_tol=1e-3,  # optimizer tolerance stopping criterion
        penalty_weight=1e-3,  # ridge regularization strength (for eliminating flat directions)
        softplus_scale=1,
        optimizer="lbfgs",
        is_classifier=False,
        yhat_init=None,
        random_state=None,
    ):
        self.n_boosts = n_boosts
        self.optim_maxiter = optim_maxiter
        self.optim_tol = optim_tol
        # ridge regularization strength
        self.penalty_weight = penalty_weight
        self.softplus_scale = softplus_scale

        if optimizer == "gd":
            self.optimizer = GradientDescent
        if optimizer == "lbfgs":
            self.optimizer = LBFGS

        if is_classifier:
            self.loss = loss_logistic
        else:
            self.loss = loss_quadratic

        self.is_classifier = is_classifier

        # boosting from yhat_init
        self.yhat_init = yhat_init

        # random seed for reproducibility
        self.random_state = random_state

        # train losses for each boosting iteration
        self.losses = None
        # model parameters
        self.params = None

    def fit(self, x, y):
        x_with_intercept = add_intercept(x)

        if self.random_state is not None:
            key = jax.random.PRNGKey(self.random_state)
        else:
            # random seed not set, get a random seed (very hacky)
            # TODO find if there is a better way to "not set a seed"
            key = jax.random.PRNGKey(np.random.randint(low=0, high=1e16))

        # learn
        params, losses = learn(
            m=self.n_boosts,
            x=x_with_intercept,
            y=y,
            key=key,
            optimizer_params={"maxiter": self.optim_maxiter, "tol": self.optim_tol},
            ridge=self.penalty_weight,
            y0=self.yhat_init,
            loss=self.loss,
            softplus_scale=self.softplus_scale,
            optimizer=self.optimizer,
        )
        self.params = params
        self.losses = losses

    def predict(self, x, y0=0.0, n_boosts=None, get_score=False):
        if n_boosts is None:
            n_boosts = self.n_boosts

        x_with_intercept = add_intercept(x)

        z = jnp.array(
            [
                predict1(parami, x_with_intercept, self.softplus_scale)
                for parami in self.params
            ]
        ).T

        score = y0 + jnp.sum(z, axis=1)

        if self.is_classifier:
            if get_score:
                return score
            else:
                # transform scores to labels
                return score_to_labels(score)
        else:
            return score

    def transform(self, x):
        return np.hstack(
            [
                predict1(
                    params=self.params[i], x=add_intercept(x), scale=self.softplus_scale
                )[:, None]
                for i in range(self.n_boosts)
            ]
        )

    def get_coordinate_distance(
        self, reference_points, sample_points, metric="euclidean", stat="min"
    ):
        """Compute the distance between each sample point to the set of reference points in the embedding coornidates

        Args:
            reference_points (array like): a set of reference points
            sample_points (array like): a set of sample points we want to compute the distance to reference_points

        Returns:
            array like: distance in the embedding coordinates
        """
        coor_ref = np.array(self.transform(reference_points))
        coor_sam = np.array(self.transform(sample_points))
        C = distance.cdist(coor_sam, coor_ref, metric=metric)
        if stat == "min":
            dis = np.min(C, axis=1)
        elif stat == "mean":
            dis = np.mean(C, axis=1)
        else:
            raise ValueError("stat not implemented.")

        return dis

    # get activated hyperplane
    def get_activated_hyperplane(self, x):
        x = add_intercept(x)
        m = len(self.params)
        hyperplane_idx = []
        for j in range(m):
            _, _, w = self.params[j]
            if jnp.dot(x, w.T) > 0:
                hyperplane_idx.append(j)

        return hyperplane_idx

    def get_explaination(self, x):
        activated_hyperplane = self.get_activated_hyperplane(x)
        w_score = jnp.zeros(x.shape[0] + 1)
        for i in activated_hyperplane:
            _, b, w = self.params[i]
            w_score = b * w

        return w_score
