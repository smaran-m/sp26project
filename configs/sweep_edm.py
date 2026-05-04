"""D = inf (EDM baseline). pfgmpp=False; aug_dim is unused but train.py still requires
a value (its IntRange validator demands min=2), so we set it to 2 as a no-op."""

CONFIG = dict(
    name="sweep_edm",
    pfgmpp=False,
    aug_dim=2,
)
