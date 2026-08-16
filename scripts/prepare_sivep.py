"""Prepara registros do SIVEP/OpenDataSUS para o PatientDB da Fase 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.assistant import config
from src.assistant.knowledge.sivep_adapter import (
    prepare_sqlite_from_prepared_csv,
    prepare_sqlite_from_sivep,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="CSV/Parquet anual ou diretório com INFLUD*.csv")
    ap.add_argument("--sample", action="store_true", help="usa a amostra anonimizada versionada")
    ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--output", default=str(config.SIVEP_DB_PATH))
    args = ap.parse_args()

    output = Path(args.output)
    if args.sample or not args.paths:
        total = prepare_sqlite_from_prepared_csv(config.SIVEP_SAMPLE_PATH, output)
    else:
        total = prepare_sqlite_from_sivep(args.paths, output, args.max_rows_per_file)
    print(f"[ok] {total:,} registros preparados em {output}")


if __name__ == "__main__":
    main()
