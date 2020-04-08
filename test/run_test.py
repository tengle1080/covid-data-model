import pathlib
from click.testing import CliRunner
import run_model


def test_run_model(tmpdir):

    runner = CliRunner()
    runner.invoke(
        run_model.main,
        ['county', '--output-dir', tmpdir, '--state', 'MA'],
        catch_exceptions=False
    )

    output_dir = pathlib.Path(tmpdir)
    paths = list(output_dir.iterdir())
    print(paths)

    assert 0
