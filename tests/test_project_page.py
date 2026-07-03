import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = ROOT / "docs"
INDEX = PAGE_ROOT / "index.html"
STYLE = PAGE_ROOT / "style.css"
ASSETS = PAGE_ROOT / "assets"

EXPECTED_RESOURCE_LABELS = [
    "Paper",
    "OpenReview",
    "Code coming soon",
    "Hugging Face coming soon",
    "Data coming soon",
]

REQUIRED_ASSETS = [
    "fuse-paper.pdf",
    "method-pipeline.svg",
    "sbibm-benchmark.png",
    "beta-pictoris.png",
    "best-of-n.png",
    "project-page-qr.svg",
]

PROJECT_PAGE_QR_SHA256 = "769df801045661f75a999930f37e539e5c6f7b943ef8a4d292b04791d0fe56da"

PRIVATE_MARKERS = [
    "/Users/",
    "/2024133006/",
    "mbhb_train",
    "Author Console",
    "referrer=",
]


def read_text(path):
    return path.read_text(encoding="utf-8")


def normalize_html_text(fragment):
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(fragment.split())


def resource_block(html):
    match = re.search(
        r'<div class="resource-links"[^>]*>(?P<body>.*?)</div>',
        html,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError("resource-links block is missing")
    return match.group("body")


def resource_items(block):
    pattern = re.compile(
        r'<(?P<tag>a|span)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)>',
        re.DOTALL,
    )
    return [
        {
            "tag": match.group("tag"),
            "attrs": match.group("attrs"),
            "label": normalize_html_text(match.group("body")),
        }
        for match in pattern.finditer(block)
    ]


def attr_value(attrs, name):
    match = re.search(rf'{name}="([^"]+)"', attrs)
    return match.group(1) if match else None


class ProjectPageTests(unittest.TestCase):
    def test_required_files_exist(self):
        self.assertTrue(INDEX.is_file(), "docs/index.html is missing")
        self.assertTrue(STYLE.is_file(), "docs/style.css is missing")
        for filename in REQUIRED_ASSETS:
            with self.subTest(filename=filename):
                self.assertTrue((ASSETS / filename).is_file(), f"{filename} is missing")

    def test_resource_items_match_approved_policy(self):
        html = read_text(INDEX)
        items = resource_items(resource_block(html))
        self.assertEqual([item["label"] for item in items], EXPECTED_RESOURCE_LABELS)

        hrefs = {item["label"]: attr_value(item["attrs"], "href") for item in items}
        self.assertEqual(hrefs["Paper"], "assets/fuse-paper.pdf")
        self.assertEqual(hrefs["OpenReview"], "https://openreview.net/forum?id=evIBAgZPjC")
        self.assertIsNone(hrefs["Code coming soon"])
        self.assertIsNone(hrefs["Hugging Face coming soon"])
        self.assertIsNone(hrefs["Data coming soon"])

        for item in items[2:]:
            self.assertEqual(item["tag"], "span")
            self.assertIn("is-disabled", item["attrs"])

    def test_affiliations_match_paper_order(self):
        html = read_text(INDEX)
        expected_fragments = [
            "<span>Peihao Wang<sup>2</sup></span>",
            "<span>Minghui Du<sup>3,4</sup></span>",
            "<span>Bo Liang<sup>3,4</sup></span>",
            "<span>Jiakai Zhang<sup>1,5</sup></span>",
            "<sup>1</sup>ShanghaiTech University, Shanghai, China",
            "<sup>2</sup>The University of Texas at Austin, Austin, TX, USA",
            "<sup>3</sup>Center for Gravitational Wave Experiment, National Microgravity Laboratory, Institute of Mechanics, Chinese Academy of Sciences, Beijing 100190, China",
            "<sup>4</sup>Taiji Laboratory for Gravitational Wave Universe (Beijing/Hangzhou), University of Chinese Academy of Sciences (UCAS), Beijing 100049, China",
            "<sup>5</sup>Cellverse, Co., Ltd",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, html)

        affiliation_positions = [html.index(fragment) for fragment in expected_fragments[4:]]
        self.assertEqual(affiliation_positions, sorted(affiliation_positions))

    def test_no_private_or_author_console_links_in_public_files(self):
        for path in [INDEX, STYLE]:
            text = read_text(path)
            for marker in PRIVATE_MARKERS:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, text)

    def test_local_links_are_relative_and_resolve(self):
        html = read_text(INDEX)
        links = re.findall(r'(?:src|href)="([^"]+)"', html)
        self.assertGreaterEqual(len(links), 6)
        for link in links:
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            with self.subTest(link=link):
                self.assertFalse(link.startswith("/"), link)
                self.assertTrue((PAGE_ROOT / link).is_file(), link)

    def test_images_have_alt_text_and_use_assets(self):
        html = read_text(INDEX)
        images = re.findall(r'<img\s+([^>]+)>', html)
        self.assertEqual(len(images), 5)
        for attrs in images:
            src = attr_value(attrs, "src")
            alt = attr_value(attrs, "alt")
            self.assertIsNotNone(src, attrs)
            self.assertIsNotNone(alt, attrs)
            self.assertTrue(alt.strip(), attrs)
            self.assertTrue(src.startswith("assets/"), src)
            self.assertNotIn("screenshot", src.lower())
        self.assertIn('src="assets/method-pipeline.svg"', html)

    def test_qr_code_is_stable_for_canonical_project_url(self):
        html = read_text(INDEX)
        self.assertIn("https://qinwch.github.io/FUSE/", html)
        qr_bytes = (ASSETS / "project-page-qr.svg").read_bytes()
        self.assertEqual(hashlib.sha256(qr_bytes).hexdigest(), PROJECT_PAGE_QR_SHA256)

    def test_required_public_copy_is_present(self):
        html = read_text(INDEX)
        for text in [
            "Author camera-ready version",
            "arXiv/PMLR links will be added once public.",
            "FUSE without FK-steering",
            "test-time",
            "SBIBM Benchmark",
            "beta Pictoris b",
            "Naive Best-of-N",
        ]:
            with self.subTest(text=text):
                self.assertIn(text, html)

    def test_css_contains_responsive_and_disabled_states(self):
        css = read_text(STYLE)
        self.assertIn("@media (max-width: 720px)", css)
        self.assertIn(".resource-link.is-disabled", css)
        self.assertIn("pointer-events: none", css)
        self.assertNotIn("letter-spacing: -", css)
        self.assertNotIn("vw", css)


if __name__ == "__main__":
    unittest.main()
