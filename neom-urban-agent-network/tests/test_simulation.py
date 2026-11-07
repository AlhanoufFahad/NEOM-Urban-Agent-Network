import unittest
from unittest.mock import patch, MagicMock
import sys
import types
from neom.model import CitySimulation, Config

class TestSimulation(unittest.TestCase):
    def test_has_five_agents(self):
        sim = CitySimulation()
        self.assertEqual(len(sim.agents), 5)

    def test_charging_negotiation_messages(self):
        sim = CitySimulation("charging", Config(ev_arrivals=0, randomness=0, max_steps=12))
        sim.run()
        kinds = {m.performative for m in sim.messages}
        self.assertTrue({"REQUEST", "CFP", "PROPOSE", "ACCEPT"}.issubset(kinds))
        self.assertGreaterEqual(sim.completed_conflicts, 1)
        self.assertGreaterEqual(sim.districts["D2"].parked, 1)

    def test_waste_collection(self):
        sim = CitySimulation("waste", Config(ev_arrivals=0, bin_fill_rate=0, max_steps=12))
        sim.run()
        self.assertEqual(sim.districts["D3"].bin_fill, 0)
        self.assertTrue(any("Collected waste in D3" in event for event in sim.events))

    def test_failure_safe_mode_recovery(self):
        sim = CitySimulation("failure", Config(max_steps=20, randomness=0))
        for _ in range(8): sim.step()
        self.assertTrue(any("Safe mode activated" in event for event in sim.events))
        self.assertTrue(sim.safe_mode)
        for _ in range(10): sim.step()
        self.assertFalse(sim.safe_mode)
        self.assertEqual(sim.agents["Energy"].coordinator_view, "ACTIVE")

    def test_delay_bounds(self):
        sim = CitySimulation("charging")
        offer = sim._energy_offer({"district": "D2", "count": 999, "grid_load": 100})
        self.assertLessEqual(offer["count"], sim.districts["D2"].queue)
        self.assertLessEqual(offer["steps"], sim.config.max_delay_steps)

class TestLLMIntegration(unittest.TestCase):
    @patch("neom.model.llm.propose_charging_delay")
    @patch("neom.model.llm.configured", return_value=True)
    def test_real_llm_result_is_used_and_clamped(self, _, mock_propose):
        mock_propose.return_value = {"count": 999, "steps": 999, "reasoning": "Peak warning"}
        sim = CitySimulation("charging")
        offer = sim._energy_offer({"district": "D2", "count": 5, "grid_load": 90, "congestion": 75})
        mock_propose.assert_called_once()
        self.assertEqual(offer["policy"], "llm_validated")
        self.assertEqual(offer["count"], 5)
        self.assertEqual(offer["steps"], sim.config.max_delay_steps)

    @patch("neom.model.llm.propose_charging_delay", side_effect=ConnectionError("offline"))
    @patch("neom.model.llm.configured", return_value=True)
    def test_llm_error_falls_back(self, _, mock_propose):
        sim = CitySimulation("charging")
        offer = sim._energy_offer({"district": "D2", "count": 5, "grid_load": 90})
        self.assertEqual(offer["policy"], "bounded_heuristic")
        self.assertIn("fallback", sim.last_llm_result)

    @patch("neom.model.llm.propose_charging_delay")
    def test_llm_disabled_never_calls_provider(self, mock_propose):
        sim = CitySimulation("charging", Config(llm_enabled=False))
        offer = sim._energy_offer({"district": "D2", "count": 5, "grid_load": 90})
        mock_propose.assert_not_called()
        self.assertEqual(offer["policy"], "bounded_heuristic")

if __name__ == "__main__":
    unittest.main()


class TestGroqKeyConfiguration(unittest.TestCase):
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_groq_key(self):
        from neom.llm import configured
        self.assertTrue(configured())

    @patch.dict("os.environ", {"GROQ_API_KEY": ""})
    def test_missing_key(self):
        from neom.llm import configured
        self.assertFalse(configured())


class TestModelErrorMessages(unittest.TestCase):
    def test_default_model(self):
        self.assertEqual(Config().llm_model, "openai/gpt-oss-20b")

    def test_missing_model_is_explained(self):
        from neom.llm import explain_error
        self.assertIn("not available", explain_error(RuntimeError("HTTP 404"), "openai/gpt-oss-20b"))

    def test_quota_is_explained(self):
        from neom.llm import explain_error
        self.assertIn("rate/quota", explain_error(RuntimeError("HTTP 429 quota exceeded"), "openai/gpt-oss-20b"))

    def test_exception_message_does_not_echo_secret(self):
        from neom.llm import explain_error
        self.assertNotIn("PRIVATE_SECRET", explain_error(ValueError("PRIVATE_SECRET"), "openai/gpt-oss-20b"))

class TestPredictiveAI(unittest.TestCase):
    def test_ml_model_predicts_non_negative_queue(self):
        from neom.prediction import predict_queue, trained_model
        estimate = predict_queue([12, 15, 18], 3)
        self.assertGreaterEqual(estimate, 0)
        self.assertLess(trained_model()[1], 10)

    def test_ml_in_agent_and_inform_alert(self):
        sim = CitySimulation("charging", Config(ml_prediction_enabled=True, llm_enabled=False, max_steps=5))
        sim.step()
        self.assertIn("Random Forest ML", sim.peak_predictions["D2"]["model"])
        self.assertTrue(any(m.sender == "Peak Prediction" and m.content.get("topic") == "peak_alert" for m in sim.messages))

    def test_toggle_reverts_to_trend(self):
        sim = CitySimulation("charging", Config(ml_prediction_enabled=False, llm_enabled=False))
        sim.step()
        self.assertEqual(sim.peak_predictions["D2"]["model"], "trend heuristic")

    @patch("neom.model.prediction.predict_queue", side_effect=RuntimeError("missing model"))
    def test_predictor_failure_falls_back_safely(self, _):
        sim = CitySimulation("charging", Config(ml_prediction_enabled=True, llm_enabled=False))
        sim.step()
        self.assertEqual(sim.peak_predictions["D2"]["model"], "trend heuristic")
        self.assertTrue(any("Peak ML unavailable" in event for event in sim.events))

class TestGroqRetries(unittest.TestCase):
    def setUp(self):
        import neom.llm as llm
        llm._quota_blocked_until = 0.0
        self.fake_module = types.ModuleType("langchain_groq")
        self.fake_module.ChatGroq = MagicMock()
        patcher = patch.dict(sys.modules, {"langchain_groq": self.fake_module})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.context = {"district": "D2", "queue": 20, "requested_count": 5,
                        "grid_load": 90, "peak_threshold": 80, "congestion": 75,
                        "congestion_limit": 70, "max_delay_steps": 6}

    @patch("neom.llm.time.sleep")
    @patch("neom.llm.configured", return_value=True)
    def test_504_retry_then_success(self, _, sleep):
        from neom.llm import propose_charging_delay
        model = self.fake_module.ChatGroq
        model.return_value.invoke.side_effect = [
            RuntimeError("HTTP 504: timeout"),
            types.SimpleNamespace(content='{"count": 5, "steps": 3, "reasoning": "Peak"}')]
        offer = propose_charging_delay(self.context)
        self.assertEqual(offer["steps"], 3)
        self.assertEqual(model.return_value.invoke.call_count, 2)
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(model.call_args.kwargs["max_retries"], 0)

    @patch("neom.llm.time.sleep")
    @patch("neom.llm.configured", return_value=True)
    def test_429_no_retry_and_cooldown(self, _, sleep):
        from neom.llm import propose_charging_delay, explain_error
        model = self.fake_module.ChatGroq
        model.return_value.invoke.side_effect = RuntimeError("HTTP 429 quota exceeded")
        with self.assertRaises(RuntimeError) as error:
            propose_charging_delay(self.context)
        self.assertIn("429", explain_error(error.exception, "openai/gpt-oss-20b"))
        with self.assertRaises(RuntimeError):
            propose_charging_delay(self.context)
        self.assertEqual(model.return_value.invoke.call_count, 1)
        sleep.assert_not_called()

    @patch("neom.llm.time.sleep")
    @patch("neom.llm.configured", return_value=True)
    def test_403_no_retry(self, _, sleep):
        from neom.llm import propose_charging_delay
        model = self.fake_module.ChatGroq
        model.return_value.invoke.side_effect = RuntimeError("HTTP 403")
        with self.assertRaises(RuntimeError):
            propose_charging_delay(self.context)
        self.assertEqual(model.return_value.invoke.call_count, 1)
        sleep.assert_not_called()

    @patch("neom.llm.configured", return_value=True)
    def test_non_json_fails_safe(self, _):
        from neom.llm import propose_charging_delay
        model = self.fake_module.ChatGroq
        model.return_value.invoke.return_value = types.SimpleNamespace(content="not json")
        with self.assertRaises(ValueError):
            propose_charging_delay(self.context)

    def test_json_validation(self):
        from neom.llm import _parse_proposal
        self.assertEqual(_parse_proposal('```json\n{"count": 2, "steps": 6, "reasoning": "test"}\n```')["count"], 2)
        with self.assertRaises(Exception):
            _parse_proposal('{"count": 5, "steps": 0, "reasoning": "wrong"}')
