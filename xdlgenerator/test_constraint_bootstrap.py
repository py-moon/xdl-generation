from constraint_bootstrap import ConstraintBootstrapTrainer


def test_constraint_bootstrap_learns_and_stops_on_stable_accuracy():
    calls = {"suggest": 0}

    def fake_generate_xdl(description, constraints_text):
        if "必须包含vessel" in constraints_text:
            return "<XDL><Synthesis><Hardware><Component id='beaker' type='vessel'/></Hardware><Reagents><Reagent name='water'/></Reagents><Procedure><Add vessel='beaker' reagent='water'/></Procedure></Synthesis></XDL>"
        return "<XDL><Synthesis><Hardware><Component id='beaker' type='vessel'/></Hardware><Reagents><Reagent name='water'/></Reagents><Procedure><Add reagent='water'/></Procedure></Synthesis></XDL>"

    def fake_verify_xdl(xdl):
        if "vessel='beaker'" in xdl:
            return []
        return [{"step": "<Add reagent='water'/>", "errors": ["You must have 'vessel' property when doing 'Add'"]}]

    def fake_suggest_constraints(_payload):
        calls["suggest"] += 1
        return ["Add 步骤必须包含vessel参数", "必须包含vessel"]

    trainer = ConstraintBootstrapTrainer(
        generate_xdl_fn=fake_generate_xdl,
        suggest_constraints_fn=fake_suggest_constraints,
        verify_xdl_fn=fake_verify_xdl,
    )

    result = trainer.train(
        samples=[{"description": "向烧杯中加入水"}],
        rounds=6,
        stable_window=2,
        stable_tolerance=0.0,
        min_rounds=2,
    )

    assert result["constraints"]
    assert "必须包含vessel" in result["constraints"]
    assert result["accuracy_history"][0] == 0.0
    assert result["accuracy_history"][1] == 1.0
    assert result["is_stable"] is True
    assert calls["suggest"] == 1
