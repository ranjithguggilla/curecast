"""Gradio front end for curecast.

Two tabs: score a mix you type in, and ask the optimiser for the greenest mix
that hits a strength target. The interface itself holds no logic worth
testing; it forwards to ``score_mix`` and ``optimise_mix`` and formats the
result. That separation is what lets the test-suite stand in for a person
clicking around.
"""

from __future__ import annotations

import gradio as gr

from .config import co2_per_m3, cost_per_m3
from .optimize import optimise_mix, savings_vs_baseline
from .scoring import score_mix

_BASELINE = {
    "cement": 480.0, "slag": 0.0, "ash": 0.0, "water": 192.0,
    "superplastic": 4.0, "coarseagg": 968.0, "fineagg": 773.0, "age": 28.0,
}


def _score(cement, slag, ash, water, superplastic, coarseagg, fineagg, age):
    mix = {
        "cement": cement, "slag": slag, "ash": ash, "water": water,
        "superplastic": superplastic, "coarseagg": coarseagg,
        "fineagg": fineagg, "age": age,
    }
    card = score_mix(mix)
    lines = [
        f"Predicted strength: {card['predicted_strength_mpa']} MPa",
        f"Embodied carbon: {card['co2_kg_per_m3']} kg CO2e per m3",
        f"Material cost: ${card['cost_usd_per_m3']} per m3",
        f"Carbon intensity: {card['carbon_intensity_kg_per_mpa']} kg CO2e per MPa",
        "",
        "Why:",
    ]
    lines += [f"- {r}" for r in card["reasons"]]
    return "\n".join(lines)


def _optimise(target, age):
    cand = optimise_mix(target_strength=float(target), age=float(age))
    save = savings_vs_baseline(cand, _BASELINE)
    status = "meets target" if cand.meets_target else "closest achievable (target not met)"
    mix = {k: round(v, 1) for k, v in cand.mix.items()}
    lines = [
        f"Status: {status}",
        f"Predicted strength: {cand.predicted_strength:.1f} MPa",
        f"Embodied carbon: {cand.co2:.1f} kg CO2e per m3 "
        f"(vs {co2_per_m3(_BASELINE):.1f} for a high-clinker mix)",
        f"Material cost: ${cand.cost:.2f} per m3 "
        f"(vs ${cost_per_m3(_BASELINE):.2f})",
        f"Carbon saved: {save['co2_saved_kg_per_m3']} kg per m3 "
        f"({save['co2_saved_pct']}%)",
        "",
        "Suggested mix (kg/m3 unless noted):",
        f"- cement {mix['cement']}, slag {mix['slag']}, fly ash {mix['ash']}",
        f"- water {mix['water']}, superplasticiser {mix['superplastic']}",
        f"- coarse agg {mix['coarseagg']}, fine agg {mix['fineagg']}",
        f"- curing age {mix['age']} days",
    ]
    return "\n".join(lines)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="curecast") as demo:
        gr.Markdown(
            "# curecast\n"
            "Predict concrete compressive strength from a mix design, then find "
            "the lowest-carbon mix that still hits your strength target."
        )
        with gr.Tab("Score a mix"):
            with gr.Row():
                cement = gr.Number(label="Cement (kg/m3)", value=350.0)
                slag = gr.Number(label="Blast-furnace slag (kg/m3)", value=80.0)
                ash = gr.Number(label="Fly ash (kg/m3)", value=60.0)
            with gr.Row():
                water = gr.Number(label="Water (kg/m3)", value=180.0)
                superplastic = gr.Number(label="Superplasticiser (kg/m3)", value=6.0)
                age = gr.Number(label="Curing age (days)", value=28.0)
            with gr.Row():
                coarseagg = gr.Number(label="Coarse aggregate (kg/m3)", value=968.0)
                fineagg = gr.Number(label="Fine aggregate (kg/m3)", value=773.0)
            score_btn = gr.Button("Score", variant="primary")
            score_out = gr.Textbox(label="Result", lines=10)
            score_btn.click(
                _score,
                [cement, slag, ash, water, superplastic, coarseagg, fineagg, age],
                score_out,
            )
        with gr.Tab("Optimise for a target"):
            with gr.Row():
                target = gr.Slider(15, 70, value=40, step=1, label="Target strength (MPa)")
                opt_age = gr.Slider(3, 90, value=28, step=1, label="Curing age (days)")
            opt_btn = gr.Button("Find greenest mix", variant="primary")
            opt_out = gr.Textbox(label="Recommendation", lines=12)
            opt_btn.click(_optimise, [target, opt_age], opt_out)
    return demo


def main():
    build_interface().launch()


if __name__ == "__main__":
    main()
