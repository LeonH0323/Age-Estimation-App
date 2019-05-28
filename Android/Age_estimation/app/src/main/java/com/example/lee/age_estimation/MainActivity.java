package com.example.lee.age_estimation;

import android.Manifest;
import android.app.ActionBar;
import android.app.Dialog;
import android.app.ProgressDialog;
import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Typeface;
import android.media.FaceDetector;
import android.net.Uri;
import android.os.Environment;
import android.os.Handler;
import android.os.Message;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.TimeUnit;

import butterknife.Bind;
import butterknife.ButterKnife;
import butterknife.OnClick;
import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import static com.example.lee.age_estimation.R.color.white;


public class MainActivity extends AppCompatActivity {

    @Bind(R.id.img)
    ImageView imgView;
    @Bind(R.id.age)
    TextView resultView;
    @Bind(R.id.predic)
    Button predic;
    @Bind(R.id.album)
    Button album;
    @Bind(R.id.camera)
    Button camera;
    @Bind(R.id.faceAging)
    Button faceAging;

    int faceNum = 0;
    String currentPhotoPath  = null;
    Uri currentUri = null;
    Uri photoUri = null;
    ProgressDialog progressOfPredict;
    ProgressDialog progressOfAging;
    AlertDialog.Builder nofaceDialog;
    final ArrayList<String> historyPicOfPrect = new ArrayList<>();
    final ArrayList<String> ageOfPredict = new ArrayList<>();
    final ArrayList<String> timeOfPredict = new ArrayList<>();

    PointF pointF;
    float eyesDistance;
    String ageStr="age";

    boolean doAging=false;
    final String[] agingLevel = new String[]{"20-30岁","30-40岁","40-50岁","50+"};

    private final static int NOFACE = 1;
    private final static int ALBUM = 1;
    private final static int CAMERA = 2;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        ButterKnife.bind(this);
        init();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.history, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()){
            case R.id.action_cart:
                Intent intent = new Intent(this,HistoryActivity.class);
//                intent.setData(currentUri);
                intent.putExtra("pic", historyPicOfPrect.toArray(new String[historyPicOfPrect.size()]));
                intent.putExtra("age", ageOfPredict.toArray(new String[ageOfPredict.size()]));
                intent.putExtra("time", timeOfPredict.toArray(new String[timeOfPredict.size()]));
//                intent.putExtra("pic",currentUri);
                startActivity(intent);
                break;
        }
        return super.onOptionsItemSelected(item);
    }

    private void init(){

//        ActionBar actionBar = getActionBar();
//        actionBar.setTitle("hhh");
        AssetManager assetManager = getAssets();
        Typeface typeface = Typeface.createFromAsset(assetManager,"FZBWJW.ttf");
        resultView.setTypeface(typeface);
        album.setTypeface(typeface);
        camera.setTypeface(typeface);
        predic.setTypeface(typeface);
//        resultView.setBackground();
//        Bitmap bitmap = null;
//        try {
//            InputStream imageStream = getAssets().open("image_init_3.png");
//            bitmap = BitmapFactory.decodeStream(imageStream);
//            imgView.setImageBitmap(bitmap);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
        imgView.setImageResource(R.drawable.image_init_5);
        initShowWaitingDialog();
        initNoFaceDialog();
    }

//    @OnClick({R.id.img})
//    public void selectImg(){
//
//        dialogChoose();
//    }
//
//    private void dialogChoose(){
//        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
//        builder.setTitle("提示");
//        builder.setMessage("选择图片来源");
//
//        builder.setPositiveButton("拍照", new DialogInterface.OnClickListener() {
//            @Override
//            public void onClick(DialogInterface dialog, int which) {
//                takeCamera();
//            }
//        });
//
//        builder.setNegativeButton("相册", new DialogInterface.OnClickListener() {
//            @Override
//            public void onClick(DialogInterface dialog, int which) {
//                pickAlbum();
//            }
//        });
//
//        builder.create().show();
//    }

    private Bitmap drawAge(String ageStr){
        Bitmap tmpBmap = null;
        try {
            tmpBmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),currentUri).copy(Bitmap.Config.ARGB_8888, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
//        float rate = tmpBmap.getWidth()/imgView.getWidth() > 1.0 ? tmpBmap.getWidth()/imgView.getWidth():imgView.getWidth()/tmpBmap.getWidth();
        float rate = tmpBmap.getWidth() > resultView.getWidth()? tmpBmap.getWidth()/imgView.getWidth(): (float) 1.0;
        Canvas canvas = new Canvas(tmpBmap);
        Paint paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(6*rate);
        paint.setTextSize(100*rate);
        canvas.drawText(ageStr,pointF.x,pointF.y - eyesDistance,paint);
        canvas.drawRect(
                (pointF.x - eyesDistance),
                (pointF.y - eyesDistance),
                (pointF.x + eyesDistance),
                (float) (pointF.y + eyesDistance * 1.5),
                paint
        );
        return tmpBmap;
    }

    private Bitmap faceDetect(){
        resultView.setText("预测中..");
        Bitmap tmpBmap = null;
        try {
             tmpBmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),currentUri)
                    .copy(Bitmap.Config.RGB_565, true);
        } catch (IOException e) {
            e.printStackTrace();
        }

        FaceDetector faceDetector = new FaceDetector(tmpBmap.getWidth(),tmpBmap.getHeight(), 1);
        FaceDetector.Face[] faces = new FaceDetector.Face[1];
        faceDetector.findFaces(tmpBmap,faces);
        if (faces.length == 0) {
            faceNum = 0;
        }else{
            faceNum = 1;
        }
        for (int i = 0; i < faces.length; i++) {
            FaceDetector.Face face = faces[i];
            Log.d("FaceDet", "Face ["+ face +"]");
            if (face == null) {
                faceNum = 0;
            }else{
                faceNum = 1;
            }
            if (face != null) {
                Canvas canvas = new Canvas(tmpBmap);
                Paint paint = new Paint();
                paint.setColor(Color.GREEN);
                paint.setStyle(Paint.Style.STROKE);
                paint.setStrokeWidth(8);

                pointF = new PointF();
                face.getMidPoint(pointF);
                eyesDistance =  face.eyesDistance();
                canvas.drawRect(
                         (pointF.x - eyesDistance),
                         (pointF.y - eyesDistance),
                         (pointF.x + eyesDistance),
                        (float) (pointF.y + eyesDistance * 1.5),
                        paint
                );
            }
        }

        return tmpBmap;
    }

    @OnClick({R.id.camera})
    public void takeCamera(){
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            Log.d("lesscoda", "run: noPermisssion");
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.CAMERA},
                    1);
        }else{
            Log.d("lesscoda", "run: getPermisssion");
            String SDState = Environment.getExternalStorageState();
            if (SDState.equals(Environment.MEDIA_MOUNTED)){
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                ContentValues values = new ContentValues();
                photoUri = this.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                intent.putExtra(MediaStore.EXTRA_OUTPUT,photoUri);

                startActivityForResult(intent, CAMERA);
            }
        }
    }
    @OnClick({R.id.album})
    public void pickAlbum(){
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, ALBUM);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK) {
            if (requestCode == ALBUM){
                Bitmap bt = null;
                Uri uri = data.getData();
                currentUri = uri;
                doAging = false;
                try {
                    bt = MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);

                    imgView.setImageBitmap(bt);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }else {
                String[] pojo = {MediaStore.Images.Media.DATA,
                        MediaStore.Images.Media.TITLE,
                        MediaStore.Images.Media.SIZE};
                Cursor cursor = getContentResolver().query(photoUri, pojo, null,null, null);
                if (cursor != null){
                    cursor.moveToFirst();
                    String picPath = cursor.getString(cursor.getColumnIndexOrThrow(pojo[0]));
                    currentUri = photoUri;
                    doAging = false;
                    Bitmap bt = BitmapFactory.decodeFile(picPath);
                    imgView.setImageBitmap(bt);
                }
            }
        }
    }


    private void initShowWaitingDialog(){
        progressOfPredict = new ProgressDialog(this);
        progressOfPredict.setTitle("正在预测您的年龄..");
        progressOfAging = new ProgressDialog(this);
        progressOfAging.setTitle("正在老化人脸..");
        progressOfAging.setCancelable(false);
        progressOfPredict.setCancelable(false);
    }

    private void initNoFaceDialog(){
        nofaceDialog = new AlertDialog.Builder(MainActivity.this);
        nofaceDialog.setTitle("没有发现人脸！");
        nofaceDialog.setMessage("试试其他图片吧！");
        nofaceDialog.setPositiveButton("确定", null);
        nofaceDialog.create();
    }
    @OnClick({R.id.faceAging})
    public void chooseAgingLevel(){
        AlertDialog alertDialog = new AlertDialog.Builder(this)
                .setTitle("选择老化程度")
                .setItems(agingLevel, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        progressOfAging.show();
                        Log.d("lihang", "onClick: "+doAging);
                        if (doAging == false){
                            Bitmap bitmap = null;
                            try {
                                bitmap = MediaStore.Images.Media.getBitmap(MainActivity.this.getContentResolver(),currentUri);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                            File file = convertBitmapToFile(bitmap);
                            OkHttpClient okHttpClient = new OkHttpClient().newBuilder()
                                    .connectTimeout(1,TimeUnit.SECONDS)
                                    .readTimeout(1,TimeUnit.SECONDS)
                                    .writeTimeout(1,TimeUnit.SECONDS)
                                    .build();
                            String url = "http://169.254.54.109:8868/faceAging";
                            RequestBody requestBody = new MultipartBody.Builder()
                                    .setType(MultipartBody.FORM)
                                    .addFormDataPart("img","myFace",RequestBody.create(MediaType.parse("image/jpg"),file))
                                    .build();
                            final Request request = new Request.Builder()
                                    .url(url)
                                    .post(requestBody)
                                    .build();
                            final Call call2 = okHttpClient.newCall(request);
                            call2.enqueue(new Callback() {
                                @Override
                                public void onFailure(Call call, IOException e) {
                                    Log.d("lihang", "onFailure: ");
                                    doAging = true;
                                }

                                @Override
                                public void onResponse(Call call, Response response) throws IOException {

                                }
                            });
                        }


                        String url2 = "http://169.254.54.109:8868/getAging?level="+(which+1);
                        final OkHttpClient okHttpClient1 = new OkHttpClient();
                        final Request request1 = new Request.Builder()
                                .url(url2)
                                .build();
                        new Thread(new Runnable() {
                            @Override
                            public void run() {
                                if (doAging == false) {
                                    SystemClock.sleep(35000);
                                }

                                okHttpClient1.newCall(request1).enqueue(new Callback() {
                                    @Override
                                    public void onFailure(Call call, IOException e) {

                                    }

                                    @Override
                                    public void onResponse(Call call, Response response) throws IOException {
                                        InputStream inputStream = response.body().byteStream();
                                        Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream);
                                        Message message = new Message();
                                        message.obj = bitmap1;
                                        message.what = 2;
                                        handler.sendMessage(message);
                                        progressOfAging.dismiss();
                                    }
                                });
                            }
                        }).start();
                    }
                }).create();
        alertDialog.setIcon(R.drawable.icon_aging);
        alertDialog.show();
    }


    public void doFaceAging(){
        progressOfAging.show();
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),currentUri);
        } catch (IOException e) {
            e.printStackTrace();
        }
        File file = convertBitmapToFile(bitmap);
        OkHttpClient okHttpClient = new OkHttpClient().newBuilder()
                .connectTimeout(1,TimeUnit.SECONDS)
                .readTimeout(1,TimeUnit.SECONDS)
                .writeTimeout(1,TimeUnit.SECONDS)
                .build();
        String url = "http://169.254.54.109:8868/faceAging";
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("img","myFace",RequestBody.create(MediaType.parse("image/jpg"),file))
                .build();
        final Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();
        final Call call2 = okHttpClient.newCall(request);
        call2.enqueue(new Callback() {
            @Override
            public void onFailure(Call call, IOException e) {
                Log.d("lihang", "onFailure: ");
            }

            @Override
            public void onResponse(Call call, Response response) throws IOException {
//                    InputStream inputStream = response.body().byteStream();
//                    Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream);
//                    Message message = new Message();
//                    message.obj = bitmap1;
//                    message.what = 2;
//                    handler.sendMessage(message);
//                    progressOfPredict.dismiss();
            }
        });

        String url2 = "http://169.254.54.109:8868/getAging";
        final OkHttpClient okHttpClient1 = new OkHttpClient();
        final Request request1 = new Request.Builder()
                .url(url2)
                .build();
         new Thread(new Runnable() {
             @Override
             public void run() {
                 SystemClock.sleep(30000);
                 okHttpClient1.newCall(request1).enqueue(new Callback() {
                     @Override
                     public void onFailure(Call call, IOException e) {

                     }

                     @Override
                     public void onResponse(Call call, Response response) throws IOException {
                         InputStream inputStream = response.body().byteStream();
                         Bitmap bitmap1 = BitmapFactory.decodeStream(inputStream);
                         Message message = new Message();
                         message.obj = bitmap1;
                         message.what = 2;
                         handler.sendMessage(message);
                         progressOfAging.dismiss();
                     }
                 });
             }
         }).start();
    }

    @OnClick({R.id.predic})
    public void onClickButton() {
        progressOfPredict.show();
        doPredict();
//        progressOfPredict.dismiss();
    }
    public void doPredict(){
        Bitmap bitmap = null;
        try {
            bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),currentUri);
        } catch (IOException e) {
            e.printStackTrace();
        }
        final Thread detectFace =  new Thread(new Runnable() {
            @Override
            public void run() {

                imgView.setImageBitmap(faceDetect());
                if (faceNum == 0) {
                    resultView.setText("没有发现人脸！");
                }
                Log.d("faceNum",":"+faceNum);
            }
        });

        File file = convertBitmapToFile(bitmap);
        OkHttpClient okHttpClient = new OkHttpClient();
        String url = "http://169.254.54.109:8868/upload";
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("img","myFace",RequestBody.create(MediaType.parse("image/jpg"),file))
                .build();
        Request request = new Request.Builder()
                .url(url)
                .post(requestBody)
                .build();

        final Call call = okHttpClient.newCall(request);
        final Thread predictThred = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    Response response = call.execute();
                    String result = response.body().string();
                    Log.d("lesscoda", "run: " + result);
                    ageStr = result.split("\\.")[0]+"."+result.split("\\.")[1].substring(0,1)+"岁";
                    resultView.setText("你看起来像"+ageStr);
                    historyPicOfPrect.add(currentUri.toString());
                    ageOfPredict.add(ageStr);
                    Date date =  new Date(System.currentTimeMillis());
                    SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm");
                    timeOfPredict.add(simpleDateFormat.format(date));
//                    imgView.setImageBitmap(drawAge(ageStr));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        });
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    detectFace.start();
                    detectFace.join();

                    if (faceNum == 1) {
                        predictThred.start();
                        predictThred.join();
                        imgView.setImageBitmap(drawAge(ageStr));
                        progressOfPredict.dismiss();
                    }else{
                        progressOfPredict.dismiss();
                        Message message = new Message();
                        message.what = NOFACE;
                        handler.sendMessage(message);
                    }
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }).start();

    }
    Handler handler = new Handler(){
        @Override
        public void handleMessage(Message msg) {
            switch (msg.what){
                case NOFACE:
                    nofaceDialog.show();
                case 2:
                    Dialog dialog = new Dialog(MainActivity.this,R.style.edit_AlertDialog_style);
                    dialog.setContentView(R.layout.acticity_showpic2);
                    ImageView imageView = dialog.findViewById(R.id.start_img);
                    imageView.setImageBitmap((Bitmap) msg.obj);
                    dialog.setCanceledOnTouchOutside(true);
                    dialog.show();
            }
        }
    };


    private File convertBitmapToFile(Bitmap bitmap){
        File file = new File(getCacheDir(),"myImg");
        try {
            file.createNewFile();

            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG,80,bos);
            byte[] bitmapdata = bos.toByteArray();

            FileOutputStream fos = new FileOutputStream(file);
            fos.write(bitmapdata);
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return file;
    }
}
