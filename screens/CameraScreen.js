import React from 'react';
import { View, TouchableOpacity, ImageBackground, Dimensions } from 'react-native';
import { Camera } from 'expo-camera';
import { NavigationEvents } from "react-navigation";
import * as Location from "expo-location"
import { Ionicons } from '@expo/vector-icons';
import * as Permissions from "expo-permissions";

export default class CameraComponent extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            hasPermission: false,
            cameraType: Camera.Constants.Type.back,
            openCamera: true,
            image: null,
            hasLocation: false,
            location: {
                latitude: null,
                longitude: null
            },
            errorMessage: ""
        };
        this.toggleCamera = this.toggleCamera.bind(this);
        this.retakeImage = this.retakeImage.bind(this);
        this.getLocation = this.getLocation.bind(this);
        this.sendImage = this.sendImage.bind(this);
    }

    async componentDidMount() {
        await this.getPermissions();
        await this.getLocation();
        this.listener = this.props.navigation.addListener('willFocus', this.getLocation);
    }

    componentWillUnmount() {
        this.listener.remove();
    }

    async getPermissions() {
        const camera_status = await Camera.requestPermissionsAsync();
        const location_status = await Permissions.askAsync(Permissions.LOCATION);
        let errorMessage = "";
        if (camera_status.granted !== "granted" || location_status.granted !== "granted")
            errorMessage = "Permissions denied";
        this.setState(state => ({
            hasPermission: camera_status.granted && location_status.granted,
            errorMessage: errorMessage
        }));
    }

    async getLocation() {
        console.log("loc");
        if(this.state.hasPermission) {
            let location = await Location.getCurrentPositionAsync({});
            let errorMessage = "";
            if (location.coords.latitude === null || location.coords.longitude === null) {
                errorMessage = "Could not get location";
            }
            this.setState(state => ({
                location: location.coords,
                hasLocation: true,
                errorMessage: errorMessage
            }));
            console.log(this.state);
        }
    }

    async takeImage() {
        if (this.camera && this.state.hasPermission) {
            let image = await this.camera.takePictureAsync({base64: true, exif: false});
            this.setState(state => ({
                image: image
            }));
            this.toggleCamera();
        }
    }

    toggleCamera() {
        this.setState(state => ({
            openCamera: true
        }))
    }

    retakeImage() {
        this.setState(state => ({
            image: null,
            openCamera: true,
        }))
    }

    async sendImage() {
        var myHeaders = new Headers();
        myHeaders.append("Content-Type", "application/json");


        let location = this.state.location.latitude + "," + this.state.location.longitude;
        var raw = JSON.stringify({"image": this.state.image.base64,"location": location});

        var requestOptions = {
            method: 'POST',
            headers: myHeaders,
            body: raw,
            redirect: 'follow'
        };

        try {
            let res = await fetch("https://fuelpriceapi.azurewebsites.net/upload/image", requestOptions);
            res = await res.text();
            this.setState(state => ({
                image: null,
                openCamera: false,
            }));
            this.props.navigation.navigate("Prices", {imageTaken: true});
        } catch(e) {
            console.log(e);
            this.setState(state => ({
                errorMessage: "Could not upload image."
            }))
        }
    }

    render() {
        if (this.state.image) {
            return(
                <View style={{ flex: 1 }}>
                    <ImageBackground
                        style={{
                            flex: 1,
                            justifyContent: "flex-end",
                            width: Dimensions.get('window').width,
                            height: Dimensions.get('window').height
                        }}
                        source={{uri: this.state.image.uri}}
                    >
                        <View style={{flex: 0.1, justifyContent: "space-around", flexDirection: "row"}}>
                            <Ionicons
                                name={"md-redo"}
                                size={50}
                                color={"white"}
                                onPress={this.retakeImage}
                            />
                            <Ionicons
                                name={"md-send"}
                                size={50}
                                color={"white"}
                                onPress={this.sendImage}
                            />
                        </View>
                    </ImageBackground>
                </View>
            )
        } else if(this.state.openCamera) {
            return(
                <View style={{ flex: 1 }}>
                    <NavigationEvents
                        onDidBlur={payload => this.setState(state => ({
                            openCamera: false,
                        }))}>
                    </NavigationEvents>
                    <Camera
                        style={{ flex: 1, justifyContent: "flex-end" }}
                        type={this.state.cameraType}
                        ref={ref => {this.camera = ref}}
                    >
                        <TouchableOpacity
                            style={{
                                flex: 0.1,
                                alignItems: 'center',
                                justifyContent: "center"
                            }}
                            onPress={() => {
                                this.takeImage();
                            }}>
                            <Ionicons
                                name={"md-radio-button-off"}
                                size={50}
                                color={"white"}
                            />
                        </TouchableOpacity>
                    </Camera>
                </View>
            )
        } else {
            return(
                <NavigationEvents
                    onWillFocus={payload => this.setState(state => ({
                        openCamera: true,
                    }))}>
                </NavigationEvents>
            )
        }
    }
}