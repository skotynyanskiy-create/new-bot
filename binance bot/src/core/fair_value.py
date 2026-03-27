import math
from decimal import Decimal
from typing import Dict, List

class FairValueEngine:
    def calculate_fair_value(self, underlying_price: Decimal, strike_price: Decimal) -> Decimal:
        """
        Calcola un prezzo teorico da 0.01 a 0.99 basato sulla distanza
        tra il prezzo del sottostante e lo strike price.
        """
        if strike_price <= Decimal('0'):
            return Decimal('0.50')
            
        ratio = float(underlying_price / strike_price)
        # Funzione logistica per normalizzare tra 0 e 1 la distanza
        # k = sensibilità della curva
        k = 5.0
        exponent = -k * (ratio - 1.0)
        try:
            fv_float = 1.0 / (1.0 + math.exp(exponent))
        except OverflowError:
            fv_float = 0.0 if exponent > 0 else 1.0
            
        fv = Decimal(str(round(fv_float, 4)))
        
        # Limita strettamente il floor e cap al range 0.01 - 0.99
        fv = max(Decimal('0.01'), min(Decimal('0.99'), fv))
        return fv

    def calculate_vwap(self, orderbook: Dict, depth_levels: int = 20) -> Decimal:
        """
        Calcola Volume Weighted Average Price dal orderbook.
        """
        bids = orderbook.get('bids', [])[:depth_levels]
        asks = orderbook.get('asks', [])[:depth_levels]
        
        total_volume = Decimal('0')
        total_value = Decimal('0')
        
        for price, volume in bids + asks:
            vol = Decimal(str(volume))
            prc = Decimal(str(price))
            total_volume += vol
            total_value += prc * vol
        
        if total_volume == 0:
            return Decimal('0')
        return total_value / total_volume

    def calculate_microprice(self, orderbook: Dict) -> Decimal:
        """
        Calcola microprice: VWAP dei best bid e ask.
        """
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        if not bids or not asks:
            return Decimal('0')
        
        best_bid_price, best_bid_vol = Decimal(str(bids[0][0])), Decimal(str(bids[0][1]))
        best_ask_price, best_ask_vol = Decimal(str(asks[0][0])), Decimal(str(asks[0][1]))
        
        total_vol = best_bid_vol + best_ask_vol
        if total_vol == 0:
            return (best_bid_price + best_ask_price) / 2
        return (best_bid_price * best_bid_vol + best_ask_price * best_ask_vol) / total_vol

    def detect_order_imbalance(self, orderbook: Dict, levels: int = 5) -> Decimal:
        """
        Ritorna ratio bid_volume / (bid_volume + ask_volume) per imbalance.
        >0.5 = bullish, <0.5 = bearish.
        """
        bids = orderbook.get('bids', [])[:levels]
        asks = orderbook.get('asks', [])[:levels]
        
        bid_vol = sum(Decimal(str(v)) for _, v in bids)
        ask_vol = sum(Decimal(str(v)) for _, v in asks)
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0:
            return Decimal('0.5')
        return bid_vol / total_vol
