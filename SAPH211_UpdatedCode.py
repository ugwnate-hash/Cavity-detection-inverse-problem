# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 23:13:48 2026 

@author: ugwna
"""

"""
Nonlinear Optimization for Underground Cavity Detection
Author: Nathan Ugwonali
Description: Solves an inverse problem to estimate underground cavity parameters 
             (position and size) from surface displacement measurements using 
             a custom Gauss-Newton algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors


# ==========================================
# 1. CONFIGURATION ET PHYSIQUE DU PROBLÈME
# ==========================================
class PhysicalConfig:
    """Paramètres physiques et géométriques du domaine."""
    def __init__(self):
        self.E0 = 6e10       # Module d'Young du sol
        self.Ec = 6e8        # Module d'Young de la cavité
        self.Rho0 = 2.6e3    # Densité du sol
        self.Rhoc = 2.6e3    # Densité de la cavité
        self.g = 9.81        # Gravité
        
        self.L = 100         # Largeur du domaine
        self.h = 100         # Profondeur du domaine
        self.dL = 1
        self.dh = 1

# ==========================================
# 2. MODÈLE DIRECT (FORWARD MODEL)
# ==========================================
class ForwardModel:
    """Calcule la déformation et le déplacement pour un jeu de paramètres donné."""
    def __init__(self, config: PhysicalConfig):
        self.cfg = config
        self.nx = int(config.L / config.dL)
        self.nz = int(config.h / config.dh)
        
        # Création de la grille (matrices 2D)
        self.X, self.Z = np.meshgrid(
            np.linspace(0, self.cfg.L, self.nx),
            np.linspace(0, self.cfg.h, self.nz)
        )

    def _get_material_fields(self, x_c, z_c, a, b):
        """Génère les champs de module d'Young et de densité basés sur l'ellipse."""
        domain = ((self.X - x_c) / a)**2 + ((self.Z - z_c) / b)**2 <= 1
        
        E_field = np.where(domain, self.cfg.Ec, self.cfg.E0)
        Rho_field = np.where(domain, self.cfg.Rhoc, self.cfg.Rho0)
        return E_field, Rho_field

    def compute_displacement(self, x_c, z_c, a, b):
        """
        Calcule les champs Uz et Epsilonz.
        Optimisation : Vectorisation sur l'axe des x pour supprimer une boucle for.
        """
        E_field, Rho_field = self._get_material_fields(x_c, z_c, a, b)
        
        Uz = np.zeros((self.nz, self.nx))
        Epsz = np.zeros((self.nz, self.nx))
        
        # Conditions limites du fond (z = h)
        Uz[-1, :] = 0
        
        # Intégration de bas en haut (vectorisée sur l'axe x)
        dh = self.cfg.dh
        g = self.cfg.g
        
        for i in range(self.nz - 2, 0, -1):
            force_term = 0.5 * Rho_field[i, :] * g * dh / E_field[i, :]
            Epsz[i, :] = Epsz[i+1, :] - force_term
            Uz[i, :] = Uz[i+1, :] - 0.5 * dh * Epsz[i+1, :] + force_term * dh
            
            # Conditions limites sur les bords latéraux (x=0 et x=L)
            Uz[i, 0] = 0
            Uz[i, -1] = 0

        # Conditions limites de surface (bord libre)
        Uz[0, :] = Uz[1, :]
        Uz[0, 0] = 0
        Uz[0, -1] = 0
        
        return Uz, Epsz

    def get_surface_displacement(self, params):
        """Retourne uniquement le déplacement en surface (Uz à z=0)."""
        x_c, z_c, a, b = params
        Uz, _ = self.compute_displacement(x_c, z_c, a, b)
        return Uz[0, :]

# ==========================================
# 3. OPTIMISATION (GRADIENT CONJUGUÉ NON-LINÉAIRE)
# ==========================================
class ConjugateGradientOptimizer:
    """
    Implémentation de l'algorithme du Gradient Conjugué (Polak-Ribière)
    avec calcul du pas optimal via la Hessienne locale.
    """
    def __init__(self, model: ForwardModel, target_data: np.ndarray):
        self.model = model
        self.target_data = target_data
        self.history = []
        self.error_history = []

    def _cost(self, params):
        residuals = self.model.get_surface_displacement(params) - self.target_data
        return np.sum(residuals**2)

    def _gradient_and_hessian(self, params, delta=1.0):
        # [MÊME CODE QUE PRÉCÉDEMMENT POUR LE GRADIENT ET LA HESSIENNE]
        n = len(params)
        grad = np.zeros(n)
        H = np.zeros((n, n))

        for i in range(n):
            p_plus = np.array(params, dtype=float)
            p_minus = np.array(params, dtype=float)
            p_plus[i] += delta
            p_minus[i] -= delta
            grad[i] = (self._cost(p_plus) - self._cost(p_minus)) / (2 * delta)

        for i in range(n):
            for j in range(n):
                if i == j:
                    p_plus = np.array(params, dtype=float)
                    p_minus = np.array(params, dtype=float)
                    p_plus[i] += delta
                    p_minus[i] -= delta
                    c_plus = self._cost(p_plus)
                    c_center = self._cost(params)
                    c_minus = self._cost(p_minus)
                    H[i, i] = (c_plus - 2*c_center + c_minus) / (delta**2)
                else:
                    p_pp = np.array(params, dtype=float)
                    p_pm = np.array(params, dtype=float)
                    p_mp = np.array(params, dtype=float)
                    p_mm = np.array(params, dtype=float)
                    p_pp[i] += delta; p_pp[j] += delta
                    p_pm[i] += delta; p_pm[j] -= delta
                    p_mp[i] -= delta; p_mp[j] += delta
                    p_mm[i] -= delta; p_mm[j] -= delta
                    H[i, j] = (self._cost(p_pp) - self._cost(p_pm) - self._cost(p_mp) + self._cost(p_mm)) / (4 * delta**2)

        return grad, H

    def optimize(self, initial_guess, max_iter=100, tol=1e-4, verbose=False):
        current_params = np.array(initial_guess, dtype=float)
        self.history.append(current_params)
        
        # Initialisation : La première direction est l'opposé du gradient
        grad, H = self._gradient_and_hessian(current_params)
        direction = -grad
        
        for i in range(max_iter):
            # Calcul du pas optimal (alpha) le long de la direction conjuguée
            curvature = np.dot(direction, np.dot(H, direction))
            
            if curvature <= 1e-18:
                alpha = 1e-2  # Pas de sécurité si courbure négative ou nulle
            else:
                alpha = -np.dot(grad, direction) / curvature
            
            # Mise à jour des paramètres avec un coefficient d'amortissement (0.5)
            delta_p = 0.5 * alpha * direction
            new_params = current_params + delta_p
            
            # Clipping 
            new_params[0] = np.clip(new_params[0], 10, self.model.cfg.L - 10)
            new_params[1] = np.clip(new_params[1], 10, self.model.cfg.h - 10)
            new_params[2] = np.clip(new_params[2], 2, self.model.cfg.L / 3)
            new_params[3] = np.clip(new_params[3], 2, self.model.cfg.h / 3)
            
            # Critère d'arrêt sur le pas
            error = np.linalg.norm(delta_p) / np.linalg.norm(current_params)
            self.history.append(new_params)
            self.error_history.append(error)
            
            # Nouveaux gradient et Hessienne
            new_grad, new_H = self._gradient_and_hessian(new_params)
            
            # Calcul de Beta (Méthode de Polak-Ribière)
            grad_norm_sq = np.dot(grad, grad)
            if grad_norm_sq < 1e-16:
                break # On est exactement sur un minimum plat
                
            beta = np.dot(new_grad, new_grad - grad) / grad_norm_sq
            beta = max(0, beta) # Reset automatique de la direction si beta < 0
            
            # Mise à jour de la direction conjuguée
            direction = -new_grad + beta * direction
            
            cost = self._cost(new_params)
            if verbose:
                print(f"Iter {i+1:02d} | Cost: {cost:.4e} | Beta: {beta:.2f} | Params: {np.round(new_params, 2)}")
            
            if error < tol:
                if verbose:
                    print(f"✅ Convergence atteinte après {i+1} itérations.")
                break
                
            current_params = new_params
            grad = new_grad
            H = new_H
            
        return current_params, self._cost(current_params)

# ==========================================
# 4. EXÉCUTION (MULTI-START & VISUALISATION)
# ==========================================
if __name__ == "__main__":
    np.random.seed(1) # Pour la reproductibilité des résultats
    
    config = PhysicalConfig()
    model = ForwardModel(config)
    
    # 1. Vérité terrain
    true_params = [config.L/3, 2*config.h/3, 15, 5]
    print(f"Paramètres cibles (vérité) : {np.round(true_params, 2)}")
    Uz_exp = model.get_surface_displacement(true_params)
    
    # 2. Multi-Start Optimization (Conditions initiales aléatoires)
    nb_restarts = 10
    best_cost = float('inf')
    best_params = None
    best_optimizer = None
    
    print(f"\n Lancement de l'optimisation avec {nb_restarts} points de départ aléatoires...")
    
    for run in range(nb_restarts):
        # Génération d'un point de départ aléatoire cohérent
        random_guess = [
            np.random.uniform(20, config.L - 20),   # x aléatoire
            np.random.uniform(20, config.h - 20),   # z aléatoire
            np.random.uniform(5, 20),               # a aléatoire
            np.random.uniform(3, 10)                # b aléatoire
        ]
        
        print(f"\n--- Run {run + 1}/{nb_restarts} | Initial: {np.round(random_guess, 2)} ---")
        
        optimizer = ConjugateGradientOptimizer(model, Uz_exp)
        opt_params, final_cost = optimizer.optimize(random_guess, max_iter=50, verbose=False)
        
        print(f"📍 Résultat: Cost = {final_cost:.4e} | Params = {np.round(opt_params, 2)}")
        
        # Sauvegarde du meilleur minimum global
        if final_cost < best_cost:
            best_cost = final_cost
            best_params = opt_params
            best_optimizer = optimizer

    print("\n" + "="*50)
    print(f"MEILLEUR RÉSULTAT GLOBAL : Cost = {best_cost:.4e}")
    print(f"Paramètres optimisés : {np.round(best_params, 2)}")
    print("="*50)

    # 3. Visualisation des résultats (utilisant le meilleur run)
    plt.figure(figsize=(10, 5))
    plt.plot(best_optimizer.error_history, marker='o', linestyle='-', color='purple')
    plt.yscale('log')
    plt.xlabel('Itérations')
    plt.ylabel('Erreur relative')
    plt.title('Convergence du Gradient Conjugué (Meilleur Run)')
    plt.grid(True, which="both", ls="--")
    plt.show()
    
    # 4. Dessin des ellipses (Vérité vs Prédiction du meilleur run)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, config.L)
    ax.set_ylim(config.h, 0) 
    
    x_true, z_true, a_true, b_true = true_params
    ax.add_patch(patches.Ellipse((x_true, z_true), 2*a_true, 2*b_true, 
                 linewidth=2, edgecolor='green', facecolor='none', linestyle='--', label='Vérité'))
    
    x_pred, z_pred, a_pred, b_pred = best_params
    ax.add_patch(patches.Ellipse((x_pred, z_pred), 2*a_pred, 2*b_pred, 
                 linewidth=2, edgecolor='red', facecolor='none', label='Prédiction Optimale'))
    
    ax.set_aspect('equal')
    ax.set_xlabel('Position X')
    ax.set_ylabel('Profondeur Z')
    ax.set_title("Multi-Start : Vérité vs Meilleure Prédiction")
    ax.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()
    
   
    # ==========================================
    # 5. VISUALISATION DES LIGNES DE NIVEAU (CONTOURS) ET TRAJECTOIRE
    # ==========================================
    print("\n--- Génération des lignes de niveau (Cost landscape) ---")
    print("Calcul sur la grille en cours...")
    
    # On fixe les dimensions de la cavité (a et b) à leurs vraies valeurs
    a_fixed, b_fixed = true_params[2], true_params[3]
    
    # Création d'une grille de recherche (30x30 pour que ce soit rapide)
    x_range = np.linspace(10, config.L - 10, 30)
    z_range = np.linspace(10, config.h - 10, 30)
    X_mesh, Z_mesh = np.meshgrid(x_range, z_range)
    Cost_surface = np.zeros_like(X_mesh)
    
    # Remplissage de la matrice de coût en utilisant le meilleur optimiseur
    for i in range(X_mesh.shape[0]):
        for j in range(X_mesh.shape[1]):
            params_grid = [X_mesh[i, j], Z_mesh[i, j], a_fixed, b_fixed]
            Cost_surface[i, j] = best_optimizer._cost(params_grid)
            
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Tracé des contours avec échelle colorimétrique logarithmique
    cf = ax.contourf(X_mesh, Z_mesh, Cost_surface, levels=50, cmap='viridis', norm=colors.LogNorm())
    fig.colorbar(cf, ax=ax, label='Coût (Log Scale)')
    
    # Superposition de la trajectoire d'optimisation (pour le meilleur run)
    history_arr = np.array(best_optimizer.history)
    ax.plot(history_arr[:, 0], history_arr[:, 1], marker='.', color='white', 
            markersize=6, linestyle='-', linewidth=1.5, label='Trajectoire d\'optimisation')
    
    # Marquer le point de départ
    ax.plot(history_arr[0, 0], history_arr[0, 1], 'ws', markersize=8, label='Point de départ')
    
    # Marquer la vérité terrain (Le fond du cratère)
    ax.plot(true_params[0], true_params[1], 'r*', markersize=15, label='Vérité (Cible)')
    
    # Mise en forme du graphique
    ax.set_xlabel('Position X')
    ax.set_ylabel('Profondeur Z')
    ax.set_title(f"Lignes de niveau de la fonction de coût (a={a_fixed:.1f}, b={b_fixed:.1f})\net trajectoire du Gradient Conjugué")
    ax.invert_yaxis() # On inverse Z pour que la surface soit en haut
    ax.legend(loc='upper right')
    
    plt.show()

    # ==========================================
    # 6. VISUALISATION DES CONTOURS (Profondeur Z vs Hauteur B)
    # ==========================================
    print("\n--- Génération des contours pour la profondeur (z) et la hauteur (b) ---")
    print("Calcul sur la grille en cours...")

    # On fixe la position latérale (x) et la demi-largeur (a) à leurs vraies valeurs
    x_fixed = true_params[0]
    a_fixed = true_params[2]

    # Création d'une grille de recherche (30x30)
    z_range = np.linspace(10, config.h - 10, 30)
    b_range = np.linspace(2, config.h / 3, 30)
    Z_mesh, B_mesh = np.meshgrid(z_range, b_range)
    Cost_surface_zb = np.zeros_like(Z_mesh)

    # Remplissage de la matrice de coût en utilisant le meilleur optimiseur
    for i in range(Z_mesh.shape[0]):
        for j in range(Z_mesh.shape[1]):
            # Attention à l'ordre des paramètres : [x, z, a, b]
            params_grid = [x_fixed, Z_mesh[i, j], a_fixed, B_mesh[i, j]]
            Cost_surface_zb[i, j] = best_optimizer._cost(params_grid)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé des contours avec échelle colorimétrique logarithmique
    cf = ax.contourf(Z_mesh, B_mesh, Cost_surface_zb, levels=50, cmap='viridis', norm=colors.LogNorm())
    fig.colorbar(cf, ax=ax, label='Coût (Log Scale)')

    # Superposition de la trajectoire d'optimisation (pour le meilleur run)
    history_arr = np.array(best_optimizer.history)
    # history_arr[:, 1] correspond à l'axe Z, history_arr[:, 3] correspond à l'axe b
    ax.plot(history_arr[:, 1], history_arr[:, 3], marker='.', color='white', 
            markersize=6, linestyle='-', linewidth=1.5, label='Trajectoire')

    # Marquer le point de départ
    ax.plot(history_arr[0, 1], history_arr[0, 3], 'ws', markersize=8, label='Point de départ')

    # Marquer la vérité terrain (Le fond du cratère)
    ax.plot(true_params[1], true_params[3], 'r*', markersize=15, label='Vérité (Cible)')

    # Mise en forme du graphique
    ax.set_xlabel('Profondeur de la cavité (Z)')
    ax.set_ylabel('Demi-hauteur de la cavité (b)')
    ax.set_title(f"Couplage Profondeur/Hauteur : Lignes de niveau (x={x_fixed:.1f}, a={a_fixed:.1f})\net trajectoire d'optimisation")
    ax.legend(loc='upper right')

    plt.show()