import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from src.api import app as api
from src.api import community, limits

APPLICATION = {"email": "afet@belediye.bel.tr", "sifre": "guclu-sifre-1", "kurum_adi": "Antakya Belediyesi",
               "kurum_turu": "belediye", "yetkili_adi": "Ayşe Yılmaz", "telefon": "0326 000 00 00",
               "kanit": "https://www.antakya.bel.tr iletişim sayfasındaki afet birimi e-postası"}
POST = {"baslik": "Su dağıtımı", "metin": "Defne parkında 14:00'te içme suyu dağıtılacak.", "tur": "yardim",
        "il": "Hatay", "ilce": "Defne", "gecerlilik_saat": 6}


class CommunityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patch = mock.patch.object(community, "DB_PATH", Path(self.tmp.name) / "test.db")
        patch.start()
        self.addCleanup(patch.stop)
        community._failed_logins.clear()
        limits.reset()
        community.create_admin("admin@darg.org", "Yönetici", "admin-sifre-1")
        self.client = TestClient(api.app)
        self.admin = self.auth("admin@darg.org", "admin-sifre-1")

    def tearDown(self):
        self.tmp.cleanup()

    def auth(self, email, password):
        token = self.client.post("/hesap/giris", json={"email": email, "sifre": password}).json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def approved_org(self, trusted=False):
        self.client.post("/hesap/basvuru", json=APPLICATION)
        org_id = self.client.get("/yonetim/kurumlar?durum=beklemede", headers=self.admin).json()["kurumlar"][0]["id"]
        self.client.post(f"/yonetim/kurumlar/{org_id}", json={"karar": "onayla", "guvenilir": trusted},
                         headers=self.admin)
        return org_id, self.auth(APPLICATION["email"], APPLICATION["sifre"])

    def test_application_waits_for_approval_and_password_is_not_stored(self):
        self.assertEqual(self.client.post("/hesap/basvuru", json=APPLICATION).status_code, 201)
        self.assertEqual(self.client.post("/hesap/basvuru", json=APPLICATION).status_code, 409)
        early = self.client.post("/hesap/giris", json={"email": APPLICATION["email"], "sifre": APPLICATION["sifre"]})
        self.assertEqual(early.status_code, 403)  # not reviewed yet
        with community.connect() as db:
            stored = db.execute("SELECT password_hash FROM users WHERE email = ?", (APPLICATION["email"],)).fetchone()[0]
        self.assertNotIn(APPLICATION["sifre"], stored)
        self.assertEqual(self.client.get("/yonetim/ozet", headers=self.admin).json()["bekleyen_basvuru"], 1)

    def test_new_organisation_posts_wait_for_review_then_appear_in_the_feed(self):
        _, org = self.approved_org()
        created = self.client.post("/gonderiler", json=POST, headers=org).json()
        self.assertEqual(created["gonderi"]["durum"], "beklemede")
        self.assertEqual(self.client.get("/akis").json()["gonderiler"], [])
        queue = self.client.get("/yonetim/gonderiler", headers=self.admin).json()["gonderiler"]
        self.client.post(f"/yonetim/gonderiler/{queue[0]['id']}", json={"karar": "yayinla"}, headers=self.admin)
        feed = self.client.get("/akis?il=Hatay").json()["gonderiler"]
        self.assertEqual((feed[0]["baslik"], feed[0]["kurum_adi"], feed[0]["tur_adi"]),
                         ("Su dağıtımı", "Antakya Belediyesi", "Yardım dağıtımı"))
        self.assertEqual(self.client.get("/akis?il=İzmir").json()["gonderiler"], [])  # other province

    def test_trusted_organisation_publishes_directly_and_only_the_admin_takes_a_post_down(self):
        _, org = self.approved_org(trusted=True)
        post = self.client.post("/gonderiler", json=POST, headers=org).json()["gonderi"]
        self.assertEqual(post["durum"], "yayinda")
        for device in ["cihaz-0001", "cihaz-0001", "cihaz-0002", "cihaz-0003", "cihaz-0004", "cihaz-0005"]:
            answer = self.client.post(f"/gonderiler/{post['id']}/sikayet", json={"cihaz_id": device, "neden": "yanlış saat"})
            self.assertEqual(answer.json()["durum"], "alindi")
        self.assertEqual(len(self.client.get("/akis").json()["gonderiler"]), 1)  # many reports, still published
        reported = self.client.get("/yonetim/sikayetler", headers=self.admin).json()["gonderiler"]
        self.assertEqual(reported[0]["sikayet"], 5)  # the same device counts once
        self.client.post(f"/yonetim/gonderiler/{post['id']}", json={"karar": "kaldir", "not": "saat yanlış"},
                         headers=self.admin)
        self.assertEqual(self.client.get("/akis").json()["gonderiler"], [])
        self.assertEqual(self.client.get("/yonetim/sikayetler", headers=self.admin).json()["gonderiler"], [])

    def test_every_reported_post_reaches_the_administrator(self):
        _, org = self.approved_org(trusted=True)
        post = self.client.post("/gonderiler", json=POST, headers=org).json()["gonderi"]
        self.client.post(f"/gonderiler/{post['id']}/sikayet", json={"cihaz_id": "cihaz-0001", "neden": "adres yanlış"})
        reported = self.client.get("/yonetim/sikayetler", headers=self.admin).json()["gonderiler"]
        self.assertEqual((reported[0]["id"], reported[0]["durum"], reported[0]["sikayet"], reported[0]["sikayet_nedenleri"]),
                         (post["id"], "yayinda", 1, ["adres yanlış"]))  # still published, but visible to the admin
        self.assertEqual(self.client.get("/yonetim/ozet", headers=self.admin).json()["sikayetli_gonderi"], 1)
        self.client.post(f"/yonetim/gonderiler/{post['id']}", json={"karar": "yayinla"}, headers=self.admin)
        self.assertEqual(self.client.get("/yonetim/sikayetler", headers=self.admin).json()["gonderiler"], [])  # settled
        self.assertEqual(len(self.client.get("/akis").json()["gonderiler"]), 1)

    def test_expired_posts_leave_the_feed_and_organisations_cannot_moderate(self):
        _, org = self.approved_org(trusted=True)
        self.client.post("/gonderiler", json={**POST, "gecerlilik_saat": 1}, headers=org)
        self.assertEqual(len(self.client.get("/akis").json()["gonderiler"]), 1)
        later = community.now() + community.timedelta(hours=2)
        with mock.patch.object(community, "now", return_value=later):
            self.assertEqual(self.client.get("/akis").json()["gonderiler"], [])
        self.assertEqual(self.client.get("/yonetim/ozet", headers=org).status_code, 403)
        self.assertEqual(self.client.get("/yonetim/ozet").status_code, 401)

    def test_suspended_organisation_is_logged_out_and_wrong_passwords_are_limited(self):
        org_id, org = self.approved_org()
        self.client.post(f"/yonetim/kurumlar/{org_id}", json={"karar": "askiya_al", "not": "kanıt doğrulanamadı"},
                         headers=self.admin)
        self.assertEqual(self.client.get("/hesap/ben", headers=org).status_code, 401)
        for _ in range(community.LOGIN_MAX_FAILURES):
            self.client.post("/hesap/giris", json={"email": "admin@darg.org", "sifre": "yanlis-sifre"})
        blocked = self.client.post("/hesap/giris", json={"email": "admin@darg.org", "sifre": "admin-sifre-1"})
        self.assertEqual(blocked.status_code, 429)
        with mock.patch.object(limits, "client_address", return_value="203.0.113.7"):  # the owner, elsewhere
            owner = self.client.post("/hesap/giris", json={"email": "admin@darg.org", "sifre": "admin-sifre-1"})
        self.assertEqual(owner.status_code, 200)  # someone else's wrong passwords did not lock the owner out

    def test_applications_from_one_address_are_limited(self):
        statuses = [self.client.post("/hesap/basvuru", json={**APPLICATION, "email": f"k{i}@belediye.bel.tr"}).status_code
                    for i in range(limits.APPLICATION.per_minute + 1)]
        self.assertEqual(statuses[-1], 429)
        self.assertTrue(all(status == 201 for status in statuses[:-1]))


if __name__ == "__main__":
    unittest.main()
