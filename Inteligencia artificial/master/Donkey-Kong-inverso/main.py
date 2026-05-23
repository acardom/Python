import numpy as np
import matplotlib.pyplot as plt
from donkey_kong_env import DonkeyKongInverso
from agents import monte_carlo_control, q_learning, show_policy

def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

def main():
    env = DonkeyKongInverso()
    
    print("🤖 Entrenando Monte Carlo (On-policy)...")
    Q_mc, rew_mc, steps_mc, succ_mc = monte_carlo_control(env, num_episodes=3000)
    
    print("⚡ Entrenando Q-Learning (Off-policy)...")
    Q_ql, rew_ql, steps_ql, succ_ql = q_learning(env, num_episodes=3000)
    
    # Imprimir métricas finales solicitadas por la entrega
    print("\n" + "="*40)
    print("📈 MÉTRICAS FINALES (ÚLTIMOS 500 EPISODIOS)")
    print("="*40)
    print(f"Monte Carlo | Éxito: {np.mean(succ_mc[-500:])*100:.2f}% | Pasos medios: {np.mean(steps_mc[-500:]):.2f}")
    print(f"Q-Learning  | Éxito: {np.mean(succ_ql[-500:])*100:.2f}% | Pasos medios: {np.mean(steps_ql[-500:]):.2f}")
    print("="*40)
    
    print("\n🗺️ POLÍTICA FINAL GREEDY APRENDIDA (Q-LEARNING):")
    show_policy(Q_ql, env)
    
    # Renderizado de curvas de aprendizaje
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rew_mc), label="Monte Carlo (On-policy)", color='royalblue')
    plt.plot(moving_average(rew_ql), label="Q-Learning (Off-policy)", color='darkorange')
    plt.title("Evolución de las Recompensas (Media Móvil 100 Episodios)")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa Media Acumulada")
    plt.legend()
    plt.grid(True, linestyle='--')
    
    print("\n📊 Mostrando gráficas... Cierra la ventana de la gráfica para finalizar el script.")
    plt.show()

if __name__ == "__main__":
    main()