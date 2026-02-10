#!/usr/bin/env python3
"""
Generate advanced animations for social dynamics ODE visualization
"""

import sys
import os
os.chdir('/Users/justu/Desktop/PhD/Projects/social-dynamics-ode')
sys.path.insert(0, os.getcwd())

# Import directly from the module file
exec(open('social_dynamics_model.py').read())
# SCENARIOS and simulate_trajectory should now be available
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

print("🎬 Creating 4 Advanced Animated Visualizations")
print("=" * 80)

anim_dir = "animations"
if not os.path.exists(anim_dir):
    os.makedirs(anim_dir)

# 1. DUAL-SCENARIO COMPARISON
print("\n  1️⃣  Dual Scenario Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Love Dynamics: The Chase vs The Friend Zone", fontsize=14, fontweight='bold', y=0.98)

s1, s2 = SCENARIOS['chase'], SCENARIOS['friend_zone']
a1, b1, c1, d1 = s1['params']
a2, b2, c2, d2 = s2['params']
t1, tr1 = simulate_trajectory([3, -1], (a1, b1, c1, d1), t_max=20, num_points=400)
t2, tr2 = simulate_trajectory([3, -1], (a2, b2, c2, d2), t_max=20, num_points=400)

for ax, traj, params, title in [(ax1, tr1, (a1,b1,c1,d1), 'The Chase'), (ax2, tr2, (a2,b2,c2,d2), 'Friend Zone')]:
    limit = 4
    A_g, B_g = np.meshgrid(np.linspace(-limit, limit, 12), np.linspace(-limit, limit, 12))
    dA = params[0] * A_g + params[1] * B_g
    dB = params[2] * A_g + params[3] * B_g
    ax.quiver(A_g, B_g, dA, dB, np.sqrt(dA**2 + dB**2), cmap='coolwarm', alpha=0.5, scale=40)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('Person A Affinity', fontsize=10)
    ax.set_ylabel('Person B Affinity', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.plot(0, 0, 'r*', markersize=20)

l1 = [ax1.plot([], [], 'o-', color='purple', markersize=8, linewidth=2)[0]]
l2 = [ax2.plot([], [], 'o-', color='blue', markersize=8, linewidth=2)[0]]
t1x = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
t2x = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

def anim1_fn(frame):
    idx = frame % 400
    l1[0].set_data(tr1[:idx, 0], tr1[:idx, 1])
    l2[0].set_data(tr2[:idx, 0], tr2[:idx, 1])
    a1_v, b1_v = tr1[idx] if idx < len(tr1) else tr1[-1]
    a2_v, b2_v = tr2[idx] if idx < len(tr2) else tr2[-1]
    t1x.set_text(f"A: {a1_v:+.2f} {'💕' if a1_v > 0 else '💔'}\nB: {b1_v:+.2f} {'💕' if b1_v > 0 else '💔'}")
    t2x.set_text(f"A: {a2_v:+.2f} {'💕' if a2_v > 0 else '💔'}\nB: {b2_v:+.2f} {'💕' if b2_v > 0 else '💔'}")
    return l1 + l2 + [t1x, t2x]

anim = FuncAnimation(fig, anim1_fn, frames=400, interval=50, blit=True, repeat=True)
anim.save(os.path.join(anim_dir, 'comparison_chase_vs_friendzone.gif'), writer=PillowWriter(fps=20))
plt.close()
print("     ✓ comparison_chase_vs_friendzone.gif")

# 2. BIFURCATION SWEEP
print("\n  2️⃣  Bifurcation Parameter Sweep...")
fig, ax = plt.subplots(figsize=(12, 8))
param_sweep = np.linspace(-2, 2, 100)
scenarios_sweep = []
for pv in param_sweep:
    a, b, c, d = 0, pv, -pv, 0
    t_sim, traj = simulate_trajectory([2, 0.5], (a, b, c, d), t_max=15, num_points=300)
    if traj is not None:
        scenarios_sweep.append((pv, traj[-100:]))

def anim2_fn(frame):
    ax.clear()
    param_idx = min(frame, len(scenarios_sweep) - 1)
    for i in range(0, param_idx + 1, max(1, param_idx // 10)):
        pv, traj = scenarios_sweep[i]
        alpha = 0.3 + 0.7 * (i / max(1, param_idx))
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color=plt.cm.RdYlBu(i / len(scenarios_sweep)), alpha=alpha, markersize=3, linewidth=1)
    if param_idx < len(scenarios_sweep):
        pv, traj = scenarios_sweep[param_idx]
        ax.plot(traj[:, 0], traj[:, 1], 'o-', color='red', markersize=6, linewidth=2.5, label='Current')
        ax.set_title(f"Bifurcation: Parameter b = {pv:.2f}\n(A's response to B)", fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Person A Affinity', fontsize=11)
    ax.set_ylabel('Person B Affinity', fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.plot(0, 0, 'r*', markersize=20, label='Equilibrium')
    ax.legend(loc='upper right')
    return []

anim = FuncAnimation(fig, anim2_fn, frames=100, interval=100, blit=True, repeat=True)
anim.save(os.path.join(anim_dir, 'bifurcation_sweep.gif'), writer=PillowWriter(fps=15))
plt.close()
print("     ✓ bifurcation_sweep.gif")

# 3. STABILITY BASIN HEAT MAP
print("\n  3️⃣  Stability Basin Heat Map...")
fig, ax = plt.subplots(figsize=(10, 10))
scenario = SCENARIOS['fire_and_ice']
a, b, c, d = scenario['params']
grid_size = 40
limit = 4
x_init = np.linspace(-limit, limit, grid_size)
y_init = np.linspace(-limit, limit, grid_size)
convergence_map = np.zeros((grid_size, grid_size))

for i, y0_val in enumerate(y_init):
    for j, x0_val in enumerate(x_init):
        t_sim, traj = simulate_trajectory([x0_val, y0_val], (a, b, c, d), t_max=10, num_points=200)
        if traj is not None:
            distances = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
            idx_close = np.where(distances < 0.1)[0]
            convergence_map[i, j] = idx_close[0] if len(idx_close) > 0 else 200

def anim3_fn(frame):
    ax.clear()
    reveal_pct = (frame + 1) / 100
    reveal_row = int(grid_size * reveal_pct)
    im = ax.imshow(convergence_map[:reveal_row, :], extent=[-limit, limit, -limit, limit], origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=200)
    ax.set_xlabel('Person A Affinity', fontsize=11)
    ax.set_ylabel('Person B Affinity', fontsize=11)
    ax.set_title(f"Fire & Ice: Stability Basin ({int(reveal_pct*100)}% Complete)", fontsize=12, fontweight='bold')
    ax.plot(0, 0, 'c*', markersize=20, label='Equilibrium')
    ax.legend(loc='upper left')
    plt.colorbar(im, ax=ax, label='Convergence Time')
    return []

anim = FuncAnimation(fig, anim3_fn, frames=100, interval=50, blit=True, repeat=False)
anim.save(os.path.join(anim_dir, 'stability_basin.gif'), writer=PillowWriter(fps=20))
plt.close()
print("     ✓ stability_basin.gif")

# 4. FOUR-STORY NARRATIVE
print("\n  4️⃣  Four-Scenario Love Stories...")
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
scen_names = ['chase', 'honeymoon', 'friend_zone', 'drama_vortex']
colors_anim = ['purple', 'red', 'blue', 'orange']
axes_anim = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]

all_trajs = []
for sn in scen_names:
    s = SCENARIOS[sn]
    a, b, c, d = s['params']
    t_sim, traj = simulate_trajectory([3, -1], (a, b, c, d), t_max=15, num_points=300)
    all_trajs.append((traj, s, t_sim))

for idx, (ax, sn, col) in enumerate(zip(axes_anim, scen_names, colors_anim)):
    s = SCENARIOS[sn]
    a, b, c, d = s['params']
    limit = 4
    A_g, B_g = np.meshgrid(np.linspace(-limit, limit, 10), np.linspace(-limit, limit, 10))
    dA = a * A_g + b * B_g
    dB = c * A_g + d * B_g
    ax.quiver(A_g, B_g, dA, dB, np.sqrt(dA**2 + dB**2), cmap='coolwarm', alpha=0.4, scale=30)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('A\'s Affinity', fontsize=9)
    ax.set_ylabel('B\'s Affinity', fontsize=9)
    ax.set_title(f"📖 {s['description']}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.plot(0, 0, 'r*', markersize=16)

lines_anim = [ax.plot([], [], 'o-', color=c, markersize=8, linewidth=2.5)[0] for ax, c in zip(axes_anim, colors_anim)]
texts_anim = [ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) for ax in axes_anim]

def anim4_fn(frame):
    idx = frame % 300
    for line, text, (traj, scenario, t_sim) in zip(lines_anim, texts_anim, all_trajs):
        if idx < len(traj):
            line.set_data(traj[:idx, 0], traj[:idx, 1])
            aa, ab = traj[idx]
            mood = "🔥 Passionate!" if aa > 1 and ab > 1 else "💕 In love" if aa > 0.5 or ab > 0.5 else "😊 Content" if aa > -0.5 and ab > -0.5 else "💔 Heartbroken" if aa < -1 or ab < -1 else "😞 Uncertain"
            text.set_text(f"A: {aa:+.2f}\nB: {ab:+.2f}\n{mood}")
    return lines_anim + texts_anim

anim = FuncAnimation(fig, anim4_fn, frames=300, interval=50, blit=True, repeat=True)
anim.save(os.path.join(anim_dir, 'love_stories.gif'), writer=PillowWriter(fps=20))
plt.close()
print("     ✓ love_stories.gif")

print(f"\n✨ ADVANCED ANIMATIONS COMPLETE! ✨")
print(f"   4 new sophisticated visualizations created!")
print(f"   Check animations/ folder for all files!")
