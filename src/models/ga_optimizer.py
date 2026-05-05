"""
Genetic Algorithm hyperparameter optimization for the XGB-RF ensemble.
Uses DEAP library for evolutionary optimization.
"""

import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier


# Search space bounds
BOUNDS = [
    (50, 300),    # xgb_n_estimators
    (3, 8),       # xgb_max_depth
    (0.01, 0.3),  # xgb_learning_rate
    (0.7, 1.0),   # xgb_subsample
    (100, 400),   # rf_n_estimators
    (10, 40),     # rf_max_depth
]


def _setup_deap():
    """Set up DEAP creator classes (safe for re-import)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def ga_optimize(X_train, y_train, n_generations=30, pop_size=20,
                cv_folds=3, random_state=42):
    """Genetic Algorithm for ensemble hyperparameter optimization.

    Returns:
        best_params: dict with optimized hyperparameters
        logbook: DEAP logbook with convergence history
    """
    _setup_deap()
    random.seed(random_state)
    np.random.seed(random_state)

    toolbox = base.Toolbox()

    def create_individual():
        return [random.uniform(lo, hi) for lo, hi in BOUNDS]

    toolbox.register("individual", tools.initIterate,
                     creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        xgb_n, xgb_d, xgb_lr, xgb_sub, rf_n, rf_d = individual

        # Clamp values to bounds
        xgb_n = int(np.clip(xgb_n, *BOUNDS[0]))
        xgb_d = int(np.clip(xgb_d, *BOUNDS[1]))
        xgb_lr = np.clip(xgb_lr, *BOUNDS[2])
        xgb_sub = np.clip(xgb_sub, *BOUNDS[3])
        rf_n = int(np.clip(rf_n, *BOUNDS[4]))
        rf_d = int(np.clip(rf_d, *BOUNDS[5]))

        xgb = XGBClassifier(
            n_estimators=xgb_n, max_depth=xgb_d,
            learning_rate=xgb_lr, subsample=xgb_sub,
            eval_metric="aucpr", random_state=42,
            use_label_encoder=False, n_jobs=-1,
        )
        rf = RandomForestClassifier(
            n_estimators=rf_n, max_depth=rf_d,
            random_state=42, n_jobs=-1,
        )
        ensemble = VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf)], voting="soft", n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(
            ensemble, X_train, y_train, cv=cv, scoring="average_precision",
            n_jobs=-1,
        )
        return (scores.mean(),)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    stats.register("min", np.min)

    print(f"Starting GA optimization: {n_generations} generations, "
          f"pop_size={pop_size}")

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2,
        ngen=n_generations, stats=stats, halloffame=hof, verbose=True,
    )

    best = hof[0]
    best_params = {
        "xgb_n_estimators": int(np.clip(best[0], *BOUNDS[0])),
        "xgb_max_depth": int(np.clip(best[1], *BOUNDS[1])),
        "xgb_learning_rate": float(np.clip(best[2], *BOUNDS[2])),
        "xgb_subsample": float(np.clip(best[3], *BOUNDS[3])),
        "rf_n_estimators": int(np.clip(best[4], *BOUNDS[4])),
        "rf_max_depth": int(np.clip(best[5], *BOUNDS[5])),
    }

    print(f"\nBest AUPRC: {best.fitness.values[0]:.4f}")
    print(f"Best params: {best_params}")

    return best_params, logbook
