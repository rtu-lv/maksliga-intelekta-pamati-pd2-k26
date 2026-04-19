# 2. Praktiskais darbs

Projektu pārvaldei tiek noteiks `pyproject.toml` fails, kur ir iespējams
deklerēt dependencies, noraksturot skriptus un citas foršas lietas. Failu var
izmantot gan ar parasto pip, kā arī ar advancētākiem rīkiem kā [uv] un [poetry].

Lai nodrošinātu koda formatēšanu (UTF-8, `\n` jaunās rindas, `\t` koda ievilkums),
projekts satur [EditorConfig] konfigurāciju. Ja tav izmantotais koda redaktors to
neatbalsta pēc noklusējuma, piemēram, VSCode gadījumā, tad ir pieejamas [iespraudnes
tam nolūkam][ec-plugins].


[uv]: https://docs.astral.sh/uv
[poetry]: https://python-poetry.org

[EditorConfig]: https://editorconfig.org
[ec-plugins]: https://editorconfig.org/#download

