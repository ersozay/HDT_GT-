/**
 * Kocaman (2025) envanter doğrulaması — Sentinel-2, 2022 yaz → 2023 yaz
 * Google Earth Engine Code Editor (https://code.earthengine.google.com) betiği.
 *
 * KURULUM (bir kez):
 *  1. Assets → NEW → Shape files: gee/inventory_upload.zip içindeki dosyaları seç,
 *     asset adı: inventory_polygons
 *  2. Aynısını gee/negatives_upload.zip için yap, asset adı: negatives_sample
 *  3. Aşağıdaki iki ASSET yolunu kendi kullanıcı adınla düzelt, "Run" de.
 *  4. Tasks sekmesinde iki dışa aktarımı "RUN" ile başlat → Google Drive'a
 *     s2_polygons.csv ve s2_negatives.csv gelir (birkaç dakika).
 *  5. CSV'leri proje klasöründe outputs/gee/ altına koy ve şunu çalıştır:
 *       python3 scripts/14_sentinel_validation.py --from-gee outputs/gee
 *
 * Yöntem: SCL ile bulut/gölge/kar/su maskeli Temmuz–Ağustos medyan kompozitleri;
 * iç bölge (poligon / 40 m disk) ve 30–150 m halka için medyan NDVI ve
 * görünür parlaklık; göreli değişim = Δiç − Δhalka (Python tarafında).
 */
var INV = ee.FeatureCollection('projects/KULLANICI/assets/inventory_polygons');   // ← düzelt
var NEG = ee.FeatureCollection('projects/KULLANICI/assets/negatives_sample');     // ← düzelt
var AOI = ee.Geometry.Rectangle([35.7, 35.9, 39.25, 38.65]);

function composite(start, end) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(AOI).filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(function (im) {
      var scl = im.select('SCL');
      var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(7));          // bitki, çıplak, sınıfsız
      var ndvi = im.normalizedDifference(['B8', 'B4']).rename('ndvi');
      var bright = im.select(['B2', 'B3', 'B4']).reduce(ee.Reducer.mean())
                     .divide(10000).rename('bright');
      return ndvi.addBands(bright).updateMask(ok);
    })
    .median();
}
var pre = composite('2022-07-01', '2022-09-01');
var post = composite('2023-07-01', '2023-09-01');
var stack = pre.rename(['ndvi_pre', 'bright_pre']).addBands(post.rename(['ndvi_post', 'bright_post']));

var reducer = ee.Reducer.median().combine(ee.Reducer.count(), '', true);

function measure(fc, innerRadius) {
  return fc.map(function (f) {
    var inner = innerRadius ? f.geometry().buffer(innerRadius) : f.geometry();
    var ring = inner.buffer(150).difference(inner.buffer(30), 1);
    var a = stack.reduceRegion({reducer: reducer, geometry: inner, scale: 10, maxPixels: 1e8, bestEffort: true});
    var b = stack.reduceRegion({reducer: reducer, geometry: ring, scale: 10, maxPixels: 1e8, bestEffort: true});
    var props = {};
    ['ndvi_pre', 'ndvi_post', 'bright_pre', 'bright_post'].forEach(function (k) {
      props[k + '_in'] = a.get(k + '_median');
      props[k + '_ring'] = b.get(k + '_median');
      props[k + '_n_in'] = a.get(k + '_count');
    });
    return ee.Feature(null, props).copyProperties(f);
  });
}

Export.table.toDrive({collection: measure(INV, null), description: 's2_polygons',
                      fileFormat: 'CSV', folder: 'heyelan_gee'});
Export.table.toDrive({collection: measure(NEG, 40), description: 's2_negatives',
                      fileFormat: 'CSV', folder: 'heyelan_gee'});

Map.centerObject(AOI, 8);
Map.addLayer(post.select('ndvi').subtract(pre.select('ndvi')), {min: -0.4, max: 0.4,
             palette: ['b2182b', 'f7f7f7', '2166ac']}, 'ΔNDVI 2022→2023 yaz');
Map.addLayer(INV.style({color: 'ff0000', fillColor: '00000000', width: 1}), {}, 'Envanter');
