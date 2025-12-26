"""
IHSG Quantitative Trading Model Configuration
"""
from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class DataConfig:
    """Data fetching configuration"""
    # Stock universe - Comprehensive Liquid IHSG Universe (IDX80 + Kompas100 + Liquid Growth)
    stock_universe: List[str] = field(default_factory=lambda: [
        'FPNI.JK',
        'GOLF.JK',
        'SOFA.JK',
        'UANG.JK',
        'DSNG.JK',
        'WIIM.JK',
        'LSIP.JK',
        'WIRG.JK',
        'PGJO.JK',
        'PNLF.JK',
        'PACK.JK',
        'CDIA.JK',
        'PMUI.JK',
        'AMMN.JK',
        'TINS.JK',
        'SMIL.JK',
        'TPIA.JK',
        'CUAN.JK',
        'SCMA.JK',
        'YELO.JK',
        'RATU.JK',
        'BRPT.JK',
        'KBLV.JK',
        'HUMI.JK',
        'JMAS.JK',
        'BRMS.JK',
        'INDF.JK',
        'MAIN.JK',
        'MAPA.JK',
        'UNVR.JK',
        'BBTN.JK',
        'DOID.JK',
        'MDLA.JK',
        'PWON.JK',
        'TAPG.JK',
        'SMRA.JK',
        'BRIS.JK',
        'PNBN.JK',
        'CPRO.JK',
        'HMSP.JK',
        'KIJA.JK',
        'GZCO.JK',
        'BUKA.JK',
        'ICBP.JK',
        'MNCN.JK',
        'JPFA.JK',
        'POWR.JK',
        'HRUM.JK',
        'TOBA.JK',
        'GGRM.JK',
        'BBCA.JK',
        'MAPI.JK',
        'UNTR.JK',
        'BSDE.JK',
        'DSSA.JK',
        'PGAS.JK',
        'JSMR.JK',
        'SRTG.JK',
        'TKIM.JK',
        'BFIN.JK',
        'PGEO.JK',
        'BWPT.JK',
        'ITMG.JK',
        'BBRI.JK',
        'AKRA.JK',
        'ABMM.JK',
        'ADHI.JK',
        'AGII.JK',
        'AHAP.JK',
        'AGRO.JK',
        'ADMF.JK',
        'AISA.JK',
        'AKSI.JK',
        'ANJT.JK',
        'APEX.JK',
        'APIC.JK',
        'APLN.JK',
        'ARNA.JK',
        'ASRI.JK',
        'ASSA.JK',
        'AUTO.JK',
        'BAPA.JK',
        'BBRM.JK',
        'BCAP.JK',
        'BCIP.JK',
        'BDMN.JK',
        'BEKS.JK',
        'BGTG.JK',
        'BHIT.JK',
        'BIPI.JK',
        'BIRD.JK',
        'BJTM.JK',
        'BJBR.JK',
        'BMTR.JK',
        'BNBR.JK',
        'BNGA.JK',
        'BSSR.JK',
        'BUKK.JK',
        'CENT.JK',
        'CITA.JK',
        'CNKO.JK',
        'CTRA.JK',
        'DGIK.JK',
        'DILD.JK',
        'DMAS.JK',
        'DNAR.JK',
        'DSFI.JK',
        'ELTY.JK',
        'ELSA.JK',
        'ERTX.JK',
        'ESTI.JK',
        'GDST.JK',
        'GEMA.JK',
        'GJTL.JK',
        'GPRA.JK',
        'GSMF.JK',
        'IATA.JK',
        'IGAR.JK',
        'IKAI.JK',
        'IMAS.JK',
        'INDX.JK',
        'INPC.JK',
        'INTP.JK',
        'IPOL.JK',
        'ISSP.JK',
        'JAWA.JK',
        'ITMA.JK',
        'JKON.JK',
        'KPIG.JK',
        'LAPD.JK',
        'LEAD.JK',
        'LINK.JK',
        'LPKR.JK',
        'LPPS.JK',
        'LPPF.JK',
        'MBSS.JK',
        'MBTO.JK',
        'MDLN.JK',
        'MIKA.JK',
        'MIDI.JK',
        'MIRA.JK',
        'MLIA.JK',
        'MPMX.JK',
        'MPPA.JK',
        'MTFN.JK',
        'MTDL.JK',
        'NIRO.JK',
        'NIKL.JK',
        'NOBU.JK',
        'NISP.JK',
        'OKAS.JK',
        'PANR.JK',
        'PKPK.JK',
        'PJAA.JK',
        'PNBS.JK',
        'PSDN.JK',
        'PTBA.JK',
        'PTPP.JK',
        'PYFA.JK',
        'RALS.JK',
        'SAME.JK',
        'SDMU.JK',
        'SDPC.JK',
        'SIDO.JK',
        'SIMP.JK',
        'SMBR.JK',
        'SOCI.JK',
        'SMSM.JK',
        'SRAJ.JK',
        'SRSN.JK',
        'SSTM.JK',
        'SSMS.JK',
        'SULI.JK',
        'TAXI.JK',
        'TBIG.JK',
        'TBLA.JK',
        'TMPO.JK',
        'TMAS.JK',
        'TOTL.JK',
        'TPMA.JK',
        'TSPC.JK',
        'UNSP.JK',
        'VIVA.JK',
        'WEHA.JK',
        'WTON.JK',
        'CASA.JK',
        'DPUM.JK',
        'JGLE.JK',
        'PBSA.JK',
        'PRDA.JK',
        'CARS.JK',
        'TAMU.JK',
        'FIRE.JK',
        'WOOD.JK',
        'MPOW.JK',
        'HOKI.JK',
        'MARK.JK',
        'KIOS.JK',
        'MDKI.JK',
        'BELL.JK',
        'ZINC.JK',
        'GMFI.JK',
        'PPRE.JK',
        'WEGE.JK',
        'DWGL.JK',
        'DFAM.JK',
        'BTPS.JK',
        'PRIM.JK',
        'TRUK.JK',
        'MSIN.JK',
        'TCPI.JK',
        'RISE.JK',
        'IPCC.JK',
        'MGRO.JK',
        'MOLI.JK',
        'ANDI.JK',
        'CITY.JK',
        'SATU.JK',
        'LUCK.JK',
        'NATO.JK',
        'CLAY.JK',
        'COCO.JK',
        'JAST.JK',
        'KJEN.JK',
        'PAMG.JK',
        'BLUE.JK',
        'FUJI.JK',
        'HDIT.JK',
        'KEEN.JK',
        'GGRP.JK',
        'TFAS.JK',
        'BAPI.JK',
        'NZIA.JK',
        'OPMS.JK',
        'SLIS.JK',
        'ESIP.JK',
        'CSMI.JK',
        'BHAT.JK',
        'PGUN.JK',
        'PPGL.JK',
        'SGER.JK',
        'PNGO.JK',
        'PURI.JK',
        'HOMI.JK',
        'VICI.JK',
        'WMUU.JK',
        'BANK.JK',
        'UNIQ.JK',
        'ZYRX.JK',
        'SNLK.JK',
        'NPGF.JK',
        'ADCP.JK',
        'MGLV.JK',
        'HOPE.JK',
        'LABA.JK',
        'BMHS.JK',
        'UVCR.JK',
        'HAIS.JK',
        'GPSO.JK',
        'CMNT.JK',
        'KUAS.JK',
        'CMRY.JK',
        'OBMD.JK',
        'NASI.JK',
        'BSML.JK',
        'DRMA.JK',
        'NTBK.JK',
        'NETV.JK',
        'SMKM.JK',
        'BIKE.JK',
        'SICO.JK',
        'STAA.JK',
        'SWID.JK',
        'IBOS.JK',
        'ASHA.JK',
        'OLIV.JK',
        'CHEM.JK',
        'DEWI.JK',
        'KRYA.JK',
        'AMMS.JK',
        'ELPI.JK',
        'CRAB.JK',
        'MEDS.JK',
        'BELI.JK',
        'BSBK.JK',
        'ZATA.JK',
        'VTNY.JK',
        'ELIT.JK',
        'BEER.JK',
        'BMBL.JK',
        'WINE.JK',
        'LAJU.JK',
        'NAYZ.JK',
        'FWCT.JK',
        'VAST.JK',
        'HALO.JK',
        'BDKR.JK',
        'TRON.JK',
        'JATI.JK',
        'MPXL.JK',
        'HAJJ.JK',
        'KLAS.JK',
        'CNMA.JK',
        'RMKO.JK',
        'PPRI.JK',
        'CYBR.JK',
        'MUTU.JK',
        'RSCH.JK',
        'BABY.JK',
        'IOTF.JK',
        'AEGS.JK',
        'KOCI.JK',
        'STRK.JK',
        'PTPS.JK',
        'KOKA.JK',
        'LOPI.JK',
        'RGAS.JK',
        'MSTI.JK',
        'ASLI.JK',
        'TOSK.JK',
        'UNTD.JK',
        'SMGA.JK',
        'ALII.JK',
        'MEJA.JK',
        'LIVE.JK',
        'MPIX.JK',
        'BAIK.JK',
        'ATLA.JK',
        'AREA.JK',
        'MHKI.JK',
        'SOLA.JK',
        'PART.JK',
        'BATR.JK',
        'ISEA.JK',
        'NEST.JK',
        'DOSS.JK',
        'MDIY.JK',
        'NAIK.JK',
        'KSIX.JK',
        'HGII.JK',
        'BRRC.JK',
        'DGWG.JK',
        'MINE.JK',
        'BLOG.JK',
        'KAQI.JK',
        'FORE.JK',
        'DKHH.JK',
        'ASPI.JK',
        'AMAN.JK',
        'CARE.JK',
        'RAAM.JK',
        'MSJA.JK',
        'SMLE.JK',
        'MANG.JK',
        'PADI.JK',
        'NCKL.JK',
        'TLKM.JK',
        'DKFT.JK',
        'DATA.JK',
        'ASLC.JK',
        'AALI.JK',
        'SSIA.JK',
        'AVIA.JK',
        'BBNI.JK',
        'ANTM.JK',
        'ARTO.JK',
        'MEDC.JK',
        'WINS.JK',
        'SMDR.JK',
        'BBHI.JK',
        'CSIS.JK',
        'EMTK.JK',
        'ULTJ.JK',
        'NICL.JK',
        'OASA.JK',
        'MYOR.JK',
        'CPIN.JK',
        'CBRE.JK',
        'INDY.JK',
        'BKSL.JK',
        'BMRI.JK',
        'CLEO.JK',
        'SGRO.JK',
        'ISAT.JK',
        'ASII.JK',
        'MMLP.JK',
        'BBYB.JK',
        'BREN.JK',
        'PANI.JK',
        'HATM.JK',
        'RUIS.JK',
        'PIPA.JK',
        'CAMP.JK',
        'SILO.JK',
        'INKP.JK',
        'ESSA.JK',
        'CSRA.JK',
        'SMGR.JK',
        'MBMA.JK',
        'TOWR.JK',
        'BULL.JK',
        'AGRS.JK',
        'MTEL.JK',
        'MDKA.JK',
        'AMRT.JK',
        'BBKP.JK',
        'WOWS.JK',
        'WIFI.JK',
        'HRTA.JK',
        'ADRO.JK',
        'INCO.JK',
        'HEAL.JK',
        'FAST.JK',
        'BUMI.JK',
        'FUTR.JK',
        'MLPL.JK',
        'RAJA.JK',
        'ERAA.JK',
        'KLBF.JK',
        'IRSX.JK',
        'DEWA.JK',
        'PEGE.JK',
        'KRAS.JK',
        'REAL.JK',
        'PTRO.JK',
        'ADMR.JK',
        'ERAL.JK',
        'CGAS.JK',
        'PSAB.JK',
        'WSBP.JK',
        'ARCI.JK',
        'NSSS.JK',
        'RMKE.JK',
        'FOLK.JK',
        'EXCL.JK',
        'CBDK.JK',
        'GOTO.JK',
        'IMPC.JK',
        'INET.JK',
        'JARR.JK',
        'BEEF.JK',
        'MINA.JK',
        'BTEK.JK',
        'INDO.JK',
        'TEBE.JK',
        'GIAA.JK',
        'DOOH.JK',
        'BUVA.JK',
        'AYAM.JK',
        'FILM.JK',
        'COIN.JK',
        'VKTR.JK',
        'PSKT.JK',
        'ENRG.JK',
        'IMJS.JK',
        'TRIN.JK',
        'GTSI.JK',
        'INTA.JK',
        'BOAT.JK',
        'UDNG.JK',
        'TRUE.JK',
        'KETR.JK',
        'MORA.JK',
        'AADI.JK',
        'COAL.JK',
        'ACES.JK',
        'TUGU.JK'
    ])
    
    # Data storage
    db_path: str = "data/ihsg_trading.db"
    
    # Historical data settings
    lookback_days: int = 365 * 5  # 5 years of history
    min_data_points: int = 100  # Minimum data points for calculation


@dataclass
class SignalConfig:
    """Signal generation parameters"""
    # Technical indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Mean reversion
    zscore_period: int = 20
    zscore_entry_threshold: float = 2.0
    
    # Momentum
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20])


@dataclass
class MLConfig:
    """Machine learning model configuration"""
    # Training
    training_window: int = 252  # Rolling window size (1 year)
    validation_split: float = 0.2
    
    # Prediction target (aligned with max_holding_days for day trading)
    target_horizon: int = 5  # Predict 5-day forward return (matches max hold)
    
    # XGBoost parameters
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    min_child_weight: int = 3
    
    # Feature settings
    feature_lags: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 20])


@dataclass
class ExitConfig:
    """Exit strategy configuration"""
    # ATR-based stops
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    trailing_stop_atr_mult: float = 1.5
    
    # Time-based exit
    max_holding_days: int = 5
    
    # Fixed stops (fallback)
    max_loss_pct: float = 0.08  # 8% max loss
    min_profit_pct: float = 0.02  # Minimum 2% to consider partial exit


@dataclass
class PortfolioConfig:
    """Portfolio management configuration"""
    # Total portfolio value for sizing calculations
    total_value: float = 100_000_000  # Default 100M IDR
    
    # Position sizing
    max_positions: int = 10
    base_position_pct: float = 0.10  # 10% per position
    max_sector_exposure: float = 0.30  # 30% max per sector
    
    # Liquidity filter
    min_avg_volume: int = 1_000_000  # 1M shares minimum
    min_market_cap: float = 1e12  # 1 Trillion IDR minimum
    
    # Risk management
    max_portfolio_volatility: float = 0.25  # 25% annual vol target
    max_correlation: float = 0.7  # Avoid highly correlated positions


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    
    # Paths
    reports_dir: str = "reports"
    models_dir: str = "models"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.data.db_path) or "data", exist_ok=True)


# Global config instance
config = Config()
